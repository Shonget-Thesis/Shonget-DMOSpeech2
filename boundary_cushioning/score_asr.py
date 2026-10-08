"""Score saved controlled-experiment WAV files with Whisper-base.

This script updates per-case JSON and metrics.csv without regenerating speech.
Use --preflight to inspect the work without importing or loading Whisper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_controlled import _atomic_json, _normalize_words, _write_csv


ASR_MODEL = "openai/whisper-base"
NORMALIZATION = "NFKC + casefold + non-alphanumeric-to-space + whitespace-collapse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory containing cases/, audio/, and metrics.csv")
    parser.add_argument("--force", action="store_true",
                        help="Rescore rows that already contain WER")
    parser.add_argument("--preflight", action="store_true",
                        help="Check saved cases and audio without loading Whisper")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def _audio_path(output_dir: Path, relative: str) -> Path:
    output_dir = output_dir.resolve()
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"audio_file must be relative to output directory: {relative}")
    resolved = (output_dir / path).resolve()
    if not resolved.is_relative_to(output_dir):
        raise ValueError(f"audio_file leaves output directory: {relative}")
    return resolved


def _needs_score(row: dict, force: bool) -> bool:
    return force or row.get("wer") in (None, "")


def inspect_cases(output_dir: Path, force: bool) -> tuple[list[tuple[Path, dict]], dict]:
    case_dir = output_dir / "cases"
    paths = sorted(case_dir.glob("*.json")) if case_dir.is_dir() else []
    cases: list[tuple[Path, dict]] = []
    errors: list[str] = []
    missing_audio: list[str] = []
    rows_total = rows_to_score = rows_already_scored = 0

    if not paths:
        errors.append(f"No case JSON files found in {case_dir}")
    for case_file in paths:
        try:
            with case_file.open("r", encoding="utf-8") as stream:
                case = json.load(stream)
            gen_text = case["gen_text"]
            if not isinstance(gen_text, str) or not _normalize_words(gen_text):
                raise ValueError("gen_text has no words after normalization")
            rows = case["metrics"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("metrics is empty or invalid")
            methods: set[str] = set()
            for row in rows:
                method = row["method"]
                if method in methods:
                    raise ValueError(f"duplicate method {method}")
                methods.add(method)
                audio_file = _audio_path(output_dir, row["audio_file"])
                if not audio_file.is_file():
                    missing_audio.append(str(audio_file))
                rows_total += 1
                if _needs_score(row, force):
                    rows_to_score += 1
                else:
                    rows_already_scored += 1
            cases.append((case_file, case))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{case_file}: {exc}")

    summary = {
        "output_dir": str(output_dir),
        "case_files": len(paths),
        "valid_cases": len(cases),
        "rows": rows_total,
        "rows_to_score": rows_to_score,
        "rows_already_scored": rows_already_scored,
        "force": force,
        "asr_model": ASR_MODEL,
        "normalization": NORMALIZATION,
        "missing_audio": missing_audio,
        "errors": errors,
        "whisper_loaded": False,
    }
    return cases, summary


def score_case(case: dict, output_dir: Path, asr_pipe, jiwer_module,
               force: bool) -> bool:
    changed = False
    truth = _normalize_words(case["gen_text"])
    # Duplicate baseline WAVs have identical bytes; one ASR call suffices.
    score_cache: dict[str, tuple[str, float]] = {}
    for row in case["metrics"]:
        if not _needs_score(row, force):
            if row.get("transcript") is not None:
                audio_file = _audio_path(output_dir, row["audio_file"])
                digest = hashlib.sha256(audio_file.read_bytes()).hexdigest()
                score_cache[digest] = (str(row["transcript"]), float(row["wer"]))
            continue

        audio_file = _audio_path(output_dir, row["audio_file"])
        digest = hashlib.sha256(audio_file.read_bytes()).hexdigest()
        if digest in score_cache:
            transcript, wer = score_cache[digest]
        else:
            result = asr_pipe(str(audio_file))
            transcript = str(result.get("text", "")).strip()
            wer = float(jiwer_module.wer(truth, _normalize_words(transcript)))
            score_cache[digest] = (transcript, wer)
        row["transcript"] = transcript
        row["wer"] = wer
        row["asr_model"] = ASR_MODEL
        row["wer_normalization"] = NORMALIZATION
        changed = True
    return changed


def main() -> None:
    args = parse_args()
    cases, summary = inspect_cases(args.output_dir, args.force)
    print(json.dumps(summary, indent=2))
    if args.preflight:
        if summary["errors"] or summary["missing_audio"]:
            raise SystemExit(2)
        return
    if summary["errors"] or summary["missing_audio"]:
        raise SystemExit("Preflight failed; resolve invalid cases or missing audio")
    if summary["rows_to_score"] == 0:
        row_count = _write_csv(args.output_dir)
        print(f"All WER scores are present; rebuilt metrics.csv with {row_count} rows")
        return

    # Model imports happen only after all case and audio checks pass.
    import jiwer
    from transformers import pipeline

    asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=-1)
    changed_cases = 0
    for index, (case_file, case) in enumerate(cases, 1):
        if score_case(case, args.output_dir, asr_pipe, jiwer, args.force):
            _atomic_json(case_file, case)
            changed_cases += 1
            row_count = _write_csv(args.output_dir)
            print(f"Scored {case_file.name} ({index}/{len(cases)}); "
                  f"{row_count} metric rows", flush=True)
    row_count = _write_csv(args.output_dir)
    print(f"Complete: updated {changed_cases} cases; {row_count} rows in "
          f"{args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
