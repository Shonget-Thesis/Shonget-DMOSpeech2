"""Run a controlled first-student-correction experiment for DMOSpeech 2.

Each case computes one teacher estimate T and one student proposal S.  All
methods start from that shared state, use the same later student noise, and
decode with the same vocoder.  The duplicate baseline is a determinism check.

``disagreement`` in metrics.csv is rho = ||S - T|| / (||T|| + 1e-8), measured
over the complete model tensor.  Spectral flux is an audio diagnostic, not a
measure of the internal handover or a claim about perceptual quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_AUDIO = ROOT / "BENCHMARK_100_SAMPLES" / "data"
CSV_COLUMNS = (
    "sample_id",
    "speaker_id",
    "seed",
    "method",
    "wer",
    "speaker_similarity",
    "inference_seconds",
    "audio_seconds",
    "disagreement",
    "effective_w",
    "peak_flux",
    "mean_flux",
    "audio_file",
)
METHODS = ("baseline", "baseline_copy", "fixed", "proposed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 2002, 3003])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--teacher-grid", type=int, default=8)
    parser.add_argument("--trim-seconds", type=float, default=4.0,
                        help="Maximum reference length; 0 permits full clips")
    parser.add_argument("--fixed-w", type=float, default=0.8596)
    parser.add_argument("--asr", action="store_true",
                        help="Compute Whisper-base WER after synthesis")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate inputs without importing or loading the models")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.teacher_grid < 1:
        parser.error("--teacher-grid must be positive")
    if args.trim_seconds < 0:
        parser.error("--trim-seconds must be nonnegative")
    if not 0 <= args.fixed_w <= 1:
        parser.error("--fixed-w must be between 0 and 1")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values")
    return args


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _resolve_audio(item: dict, manifest: Path) -> Path:
    value = item.get("ref_audio_path")
    if value:
        path = Path(value)
        return (path if path.is_absolute() else manifest.parent / path).resolve()
    value = item.get("rel_flac_path")
    if value:
        return (BENCHMARK_AUDIO / value).resolve()
    raise ValueError(f"sample {item.get('sample_id')}: no ref_audio_path or rel_flac_path")


def _read_manifest(path: Path, limit: int | None) -> list[dict]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict):
        raw = raw.get("cases", raw.get("samples"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list or contain cases/samples")
    selected = raw[:limit]
    seen: set[str] = set()
    cases: list[dict] = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("Every manifest case must be an object")
        for key in ("sample_id", "speaker_id", "ref_text", "gen_text"):
            if key not in item or str(item[key]).strip() == "":
                raise ValueError(f"Manifest case lacks {key}: {item}")
        sid = str(item["sample_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
            raise ValueError(f"sample_id must be filename safe: {sid!r}")
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen.add(sid)
        case = dict(item)
        case["resolved_ref_audio"] = str(_resolve_audio(item, path))
        cases.append(case)
    return cases


def _available_physical_ram_gb() -> float | None:
    """Read available RAM without importing torch or loading any model."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullAvailPhys / 1024 ** 3, 2)
        elif Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) * 1024 / 1024 ** 3, 2)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _check_inputs(args: argparse.Namespace, cases: list[dict]) -> dict:
    missing = [case["resolved_ref_audio"] for case in cases
               if not Path(case["resolved_ref_audio"]).is_file()]
    over_trim = []
    if args.trim_seconds > 0:
        over_trim = [str(case["sample_id"]) for case in cases
                     if case.get("ref_duration") is not None
                     and float(case["ref_duration"]) > args.trim_seconds + 0.01]
    checkpoints = {
        "student": ROOT / "ckpts" / "model_85000.pt",
        "duration": ROOT / "ckpts" / "model_1500.pt",
    }
    missing_checkpoints = [str(path) for path in checkpoints.values() if not path.is_file()]
    available_ram_gb = _available_physical_ram_gb()
    student_checkpoint_gb = (round(checkpoints["student"].stat().st_size / 1024 ** 3, 2)
                             if checkpoints["student"].is_file() else None)
    memory_warning = None
    if (available_ram_gb is not None and student_checkpoint_gb is not None
            and available_ram_gb < 2 * student_checkpoint_gb):
        memory_warning = (
            "Available physical RAM is low relative to the student checkpoint. "
            "A CPU model load may page heavily or fail; preflight does not load it."
        )
    summary = {
        "manifest": str(args.manifest.resolve()),
        "cases": len(cases),
        "speakers": len({str(case["speaker_id"]) for case in cases}),
        "tasks": len(cases) * len(args.seeds),
        "seeds": args.seeds,
        "teacher_grid": args.teacher_grid,
        "trim_seconds": args.trim_seconds,
        "missing_audio": missing,
        "declared_reference_over_trim": over_trim,
        "missing_checkpoints": missing_checkpoints,
        "available_physical_ram_gb": available_ram_gb,
        "student_checkpoint_gb": student_checkpoint_gb,
        "memory_warning": memory_warning,
    }
    return summary


def _run_config(args: argparse.Namespace) -> dict:
    checkpoints = [ROOT / "ckpts" / "model_85000.pt",
                   ROOT / "ckpts" / "model_1500.pt"]
    return {
        "schema": 1,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampler_sha256": hashlib.sha256((ROOT / "src" / "infer.py").read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "checkpoint_files": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "modified_ns": path.stat().st_mtime_ns}
            for path in checkpoints
        ],
        "teacher_grid": args.teacher_grid,
        "teacher_stopping_time": 0.07,
        "student_start_step": 1,
        "sway_coefficient": -1.0,
        "eta": 1.0,
        "cfg_strength": 2.0,
        "trim_seconds": args.trim_seconds,
        "fixed_w": args.fixed_w,
        "adaptive_rule": "clip(1 - 0.26*tanh(disagreement), 0.74, 1)",
        "branch_order": "SHA256(sample_id:seed) seeded shuffle of four methods",
        "asr": args.asr,
        "asr_model": "openai/whisper-base" if args.asr else None,
        "device": args.device,
        "disagreement": "L2(S-T) / (L2(T) + 1e-8), complete tensor",
        "flux_region": "first 15 percent of generated waveform; diagnostic only",
    }


def _prepare_run_directory(output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "run_config.json"
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != config:
            raise ValueError(
                f"{config_file} belongs to a different configuration. "
                "Use a separate output directory."
            )
    else:
        _atomic_json(config_file, config)
    (output_dir / "cases").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)


def _write_csv(output_dir: Path) -> int:
    rows: list[dict] = []
    for path in sorted((output_dir / "cases").glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            case = json.load(stream)
        rows.extend(case["metrics"])
    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["seed"]),
                               METHODS.index(row["method"])))
    path = output_dir / "metrics.csv"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output_dir,
            prefix=".metrics.", suffix=".tmp", delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if row.get(key) is None else row[key]
                                 for key in CSV_COLUMNS})
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return len(rows)


def _case_is_complete(case_file: Path, output_dir: Path,
                      sample_id: object, seed: int) -> bool:
    try:
        with case_file.open("r", encoding="utf-8") as stream:
            case = json.load(stream)
        rows = case["metrics"]
        if (str(case["sample_id"]) != str(sample_id)
                or int(case["seed"]) != seed
                or len(rows) != len(METHODS)
                or {row["method"] for row in rows} != set(METHODS)):
            return False
        return all((output_dir / row["audio_file"]).is_file() for row in rows)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _sync_cuda(device: str, torch_module) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def _synthesize_branches(tts, reference_path: Path, case: dict,
                         seed: int, teacher_grid: int, fixed_w: float):
    import numpy as np
    import torch
    import torchaudio
    from f5_tts.infer.utils_infer import (
        convert_char_to_pinyin, preprocess_ref_audio_text, target_rms
    )

    device = tts.device
    _sync_cuda(device, torch)
    common_start = time.perf_counter()

    # Keep the baseline's duration and reference preprocessing sequence.
    duration_pred = tts.predict_duration(
        str(reference_path), case["gen_text"], case["ref_text"]
    )
    ref_audio, ref_text = preprocess_ref_audio_text(
        str(reference_path), case["ref_text"]
    )
    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != tts.target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, tts.target_sample_rate)(audio)
    audio = audio.to(device)

    text_list = convert_char_to_pinyin([ref_text + case["gen_text"]])
    ref_audio_len = audio.shape[-1] // tts.hop_length
    duration_total = ref_audio_len + duration_pred
    with torch.inference_mode():
        cond, text, step_cond, cond_mask, max_duration, duration_tensor = (
            tts._prepare_inputs(audio, text_list, duration_total)
        )

    # The teacher random stream and all student noise are shared by every arm.
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        teacher = tts._teacher_sampling(
            step_cond=step_cond, text=text, cond_mask=cond_mask,
            max_duration=max_duration, duration=duration_tensor,
            teacher_steps=teacher_grid, teacher_stopping_time=0.07,
            eta=1.0, cfg_strength=2.0, verbose=False,
            sway_sampling_coef=-1.0,
        )

    steps = torch.tensor([0.0, 0.25, 0.5, 0.75], device=device)
    steps = steps - (torch.cos(torch.pi / 2 * steps) - 1 + steps)
    steps = steps[1:]
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    t1 = steps[0].view(1, 1, 1)
    first_noise = torch.randn(teacher.shape, generator=rng, device=device,
                              dtype=teacher.dtype)
    phi1 = (1.0 - t1) * first_noise + t1 * teacher
    with torch.inference_mode():
        prediction = tts.generator(
            x=phi1, cond=cond, text=text, time=steps[0].view(1),
            drop_audio_cond=False, drop_text=False,
        )
        student = phi1 + (1.0 - t1) * prediction

    displacement = student - teacher
    disagreement = float(
        torch.linalg.vector_norm(displacement).item()
        / (torch.linalg.vector_norm(teacher).item() + 1e-8)
    )
    remaining_noise = [
        torch.randn(teacher.shape, generator=rng, device=device,
                    dtype=teacher.dtype)
        for _ in steps[1:]
    ]
    weights = {
        "baseline": 1.0,
        "baseline_copy": 1.0,
        "fixed": fixed_w,
        "proposed": float(np.clip(1.0 - 0.26 * np.tanh(disagreement),
                                  0.74, 1.0)),
    }
    # Rotate computation order across cases so one arm cannot systematically
    # benefit from a cold or warm CPU cache.  This does not touch torch's RNG.
    order_seed = hashlib.sha256(
        f"{case['sample_id']}:{seed}".encode("utf-8")
    ).digest()
    branch_order = list(METHODS)
    random.Random(int.from_bytes(order_seed[:8], "big")).shuffle(branch_order)
    _sync_cuda(device, torch)
    common_seconds = time.perf_counter() - common_start

    waves = {}
    times = {}
    for method in branch_order:
        _sync_cuda(device, torch)
        branch_start = time.perf_counter()
        weight = weights[method]
        # Full acceptance uses S directly, as in the unmodified baseline.
        endpoint = student if weight == 1.0 else teacher + weight * displacement
        for index, step in enumerate(steps[1:]):
            t = step.view(1, 1, 1)
            phi = (1.0 - t) * remaining_noise[index] + t * endpoint
            with torch.inference_mode():
                prediction = tts.generator(
                    x=phi, cond=cond, text=text, time=step.view(1),
                    drop_audio_cond=False, drop_text=False,
                )
                endpoint = phi + (1.0 - t) * prediction
        with torch.inference_mode():
            mel = endpoint.permute(0, 2, 1) * tts.scale
            wave = tts.vocos.decode(mel[..., cond_mask.sum():])
            waves[method] = wave.cpu().numpy().squeeze()
        _sync_cuda(device, torch)
        times[method] = common_seconds + (time.perf_counter() - branch_start)

    if not np.array_equal(waves["baseline"], waves["baseline_copy"]):
        raise AssertionError("Duplicate baseline differs from baseline; branching is not deterministic")
    return waves, weights, disagreement, times, branch_order


def _spectral_flux(wave) -> tuple[float, float]:
    import librosa
    import numpy as np

    stft = np.abs(librosa.stft(wave, n_fft=1024, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    if flux.size == 0:
        raise ValueError("Generated audio is too short for spectral flux")
    first = flux[:max(1, int(flux.size * 0.15))]
    return float(np.max(first)), float(np.mean(first))


def _normalize_words(text: str) -> str:
    # Apply the same simple case/punctuation normalization to truth and ASR.
    import unicodedata

    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(
        char if char.isalnum() or char.isspace() else " " for char in text
    ).split())


def _run_case(tts, asr_pipe, case: dict, seed: int,
              args: argparse.Namespace, output_dir: Path) -> dict:
    import librosa
    import numpy as np
    import soundfile as sf

    sid = str(case["sample_id"])
    key = f"{sid}_seed{seed}"
    with tempfile.TemporaryDirectory(prefix=f"ref_{key}_", dir=output_dir) as temp:
        audio, sr = librosa.load(case["resolved_ref_audio"], sr=24000, mono=True)
        if args.trim_seconds > 0 and len(audio) > int(args.trim_seconds * sr):
            raise ValueError(
                f"sample {sid}: reference is {len(audio)/sr:.3f}s, above the "
                f"{args.trim_seconds:g}s limit. Clipping while keeping its full "
                "transcript would misalign the prompt; select a shorter reference "
                "or use --trim-seconds 0."
            )
        normalized_ref = Path(temp) / "reference.wav"
        sf.write(normalized_ref, audio, sr)
        waves, weights, disagreement, times, branch_order = _synthesize_branches(
            tts, normalized_ref, case, seed, args.teacher_grid, args.fixed_w
        )

    metrics = []
    for method in METHODS:
        wave = waves[method]
        if not np.isfinite(wave).all():
            raise ValueError(f"Non-finite waveform for {key}/{method}")
        relative_audio = Path("audio") / f"{key}_{method}.wav"
        audio_file = output_dir / relative_audio
        sf.write(audio_file, wave, 24000)
        peak_flux, mean_flux = _spectral_flux(wave)
        transcript = None
        wer = None
        if asr_pipe is not None:
            import jiwer

            transcript = str(asr_pipe(str(audio_file)).get("text", "")).strip()
            wer = float(jiwer.wer(_normalize_words(case["gen_text"]),
                                  _normalize_words(transcript)))
        metrics.append({
            "sample_id": case["sample_id"],
            "speaker_id": case["speaker_id"],
            "seed": seed,
            "method": method,
            "wer": wer,
            "speaker_similarity": None,
            "inference_seconds": times[method],
            "audio_seconds": len(wave) / 24000,
            "disagreement": disagreement,
            "effective_w": weights[method],
            "peak_flux": peak_flux,
            "mean_flux": mean_flux,
            "audio_file": relative_audio.as_posix(),
            "transcript": transcript,
        })

    if any(metrics[0][metric] != metrics[1][metric]
           for metric in ("wer", "audio_seconds", "peak_flux", "mean_flux")):
        raise AssertionError("Duplicate baseline evaluation differs from baseline")
    return {
        "sample_id": case["sample_id"],
        "speaker_id": case["speaker_id"],
        "seed": seed,
        "ref_audio_path": case["resolved_ref_audio"],
        "ref_text": case["ref_text"],
        "gen_text": case["gen_text"],
        "branch_order": branch_order,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    cases = _read_manifest(args.manifest, args.limit)
    summary = _check_inputs(args, cases)
    print(json.dumps(summary, indent=2))
    if args.preflight:
        if summary["missing_audio"] or summary["declared_reference_over_trim"] \
                or summary["missing_checkpoints"]:
            raise SystemExit(2)
        return
    if summary["missing_audio"] or summary["declared_reference_over_trim"] \
            or summary["missing_checkpoints"]:
        raise SystemExit("Preflight failed; resolve missing assets or overlong references")

    # Heavy imports and the model load happen only after preflight succeeds.
    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from infer import DMOInference

    config = _run_config(args)
    _prepare_run_directory(args.output_dir, config)
    tts = DMOInference(
        student_checkpoint_path=str(ROOT / "ckpts" / "model_85000.pt"),
        duration_predictor_path=str(ROOT / "ckpts" / "model_1500.pt"),
        device=args.device,
        model_type="F5TTS_Base",
    )
    tts.model.eval()
    tts.teacher.eval()
    tts.generator.eval()
    tts.vocos.eval()
    asr_pipe = None
    if args.asr:
        from transformers import pipeline

        asr_pipe = pipeline("automatic-speech-recognition",
                            model="openai/whisper-base", device=-1)

    completed = 0
    for case in cases:
        for seed in args.seeds:
            case_file = args.output_dir / "cases" / f"{case['sample_id']}_seed{seed}.json"
            if case_file.exists() and _case_is_complete(
                case_file, args.output_dir, case["sample_id"], seed
            ):
                completed += 1
                continue
            print(f"Generating sample={case['sample_id']} speaker={case['speaker_id']} seed={seed}",
                  flush=True)
            result = _run_case(tts, asr_pipe, case, seed, args, args.output_dir)
            _atomic_json(case_file, result)
            count = _write_csv(args.output_dir)
            completed += 1
            print(f"Saved {case_file.name}; {completed}/{summary['tasks']} tasks, "
                  f"{count} metric rows", flush=True)
    count = _write_csv(args.output_dir)
    print(f"Complete: {completed} cases, {count} rows in {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
