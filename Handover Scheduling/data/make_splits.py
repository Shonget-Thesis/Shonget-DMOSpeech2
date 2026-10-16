"""
Task 3: Speaker-Disjoint Data Splits for Handover Scheduling
============================================================
Creates speaker-disjoint development, calibration, and test splits from Seed-TTS test-en.
- Source: data/seed-tts-eval/en/meta.lst (1,088 utterances across 666 unique speakers)
- Authoritative speaker identity: Prompt reference speaker (prompt-wavs/common_voice_en_*.wav)
- Split sizes:
    - Development: 100 speakers / 100 utterances
    - Calibration: 200 speakers / 200 utterances (exactly 1 utterance per speaker)
    - Test: 200 speakers / 200 utterances (exactly 1 utterance per speaker)
- Conditions: clean and noisy15 (15 dB white Gaussian noise relative to prompt power)
- Strict validation:
    1. Zero speaker overlap across splits
    2. At most one utterance per speaker in calibration and test
    3. Every audio path resolves to existing files
    4. Injected SNR within 15.0 +/- 0.1 dB
"""

import os
import sys
import json
import random
import math
from typing import Dict, List, Any, Tuple
import numpy as np
import soundfile as sf
import torchaudio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HANDOVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if HANDOVER_DIR not in sys.path:
    sys.path.insert(0, HANDOVER_DIR)

from core.sampler import derive_input_seed, make_local_generator

META_LST = os.path.join(ROOT_DIR, "data", "seed-tts-eval", "en", "meta.lst")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
BASE_SEED = 42
SPLIT_SEED = 42

DEV_SIZE = 100
CALIB_SIZE = 200
TEST_SIZE = 200


def add_gaussian_noise_15db(
    clean_wav: np.ndarray,
    seed: int,
    target_snr_db: float = 15.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Adds white Gaussian noise at target_snr_db (15.0 dB) relative to clean signal power.
    
    Noise equation:
        P_sig = mean(x^2)
        P_noise = P_sig / (10^(target_snr_db / 10))
        e = randn() * sqrt(P_noise)
    
    Returns:
        noisy_wav: Clean audio + noise (clipped to [-1.0, 1.0])
        noise: The exact added noise waveform
        injected_snr: Measured SNR between clean_wav and noise
    """
    rng = np.random.RandomState(seed % (2**31 - 1))
    sig_pwr = float(np.mean(clean_wav**2))
    if sig_pwr < eps:
        # Near-silent waveform fallback
        return clean_wav.copy(), np.zeros_like(clean_wav), float(target_snr_db)

    noise_pwr_target = sig_pwr / (10.0 ** (target_snr_db / 10.0))
    raw_noise = rng.randn(*clean_wav.shape).astype(np.float32)

    # Normalize raw noise to exact target variance
    curr_noise_pwr = float(np.mean(raw_noise**2)) + eps
    noise = raw_noise * np.sqrt(noise_pwr_target / curr_noise_pwr)

    # Exact injected SNR prior to clipping
    actual_noise_pwr = float(np.mean(noise**2)) + eps
    injected_snr = 10.0 * np.log10(sig_pwr / actual_noise_pwr)

    noisy_wav = np.clip(clean_wav + noise, -1.0, 1.0)
    return noisy_wav, noise, injected_snr


def load_dataset_metadata(meta_path: str) -> List[Dict[str, Any]]:
    """Loads and validates all entries in seed-tts-eval meta.lst."""
    base_dir = os.path.dirname(meta_path)
    entries = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                utt_id, prompt_text, prompt_wav_rel, gt_text = parts
                gt_wav_rel = os.path.join("wavs", f"{utt_id}.wav")
            elif len(parts) >= 5:
                utt_id, prompt_text, prompt_wav_rel, gt_text, gt_wav_rel = parts[:5]
            else:
                continue

            prompt_wav_abs = os.path.join(base_dir, prompt_wav_rel)
            gt_wav_abs = os.path.join(base_dir, gt_wav_rel)

            # Speaker ID extracted authoritatively from prompt filename
            # e.g., prompt-wavs/common_voice_en_10119832.wav -> common_voice_en_10119832
            prompt_file = os.path.basename(prompt_wav_rel)
            speaker_id = os.path.splitext(prompt_file)[0]

            entries.append({
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "prompt_text": prompt_text,
                "prompt_wav_path": prompt_wav_abs,
                "prompt_wav_rel": prompt_wav_rel,
                "gt_text": gt_text,
                "gt_wav_path": gt_wav_abs,
                "gt_wav_rel": gt_wav_rel,
                "source_line": line_num,
            })
    return entries


def create_splits(
    entries: List[Dict[str, Any]],
    dev_n: int = DEV_SIZE,
    calib_n: int = CALIB_SIZE,
    test_n: int = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates speaker-disjoint splits with at most 1 utterance per speaker in calibration & test.
    """
    # Group utterances by speaker
    speaker_to_utts: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        spk = e["speaker_id"]
        speaker_to_utts.setdefault(spk, []).append(e)

    unique_speakers = sorted(list(speaker_to_utts.keys()))
    total_spk = len(unique_speakers)
    required_spk = dev_n + calib_n + test_n

    if total_spk < required_spk:
        raise ValueError(
            f"Not enough unique speakers! Found {total_spk}, required {required_spk}."
        )

    # Deterministic shuffle of speakers
    rng = random.Random(seed)
    shuffled_speakers = list(unique_speakers)
    rng.shuffle(shuffled_speakers)

    dev_speakers = set(shuffled_speakers[:dev_n])
    calib_speakers = set(shuffled_speakers[dev_n : dev_n + calib_n])
    test_speakers = set(shuffled_speakers[dev_n + calib_n : dev_n + calib_n + test_n])

    dev_entries = []
    calib_entries = []
    test_entries = []

    # Assign exactly one representative utterance per speaker for calibration and test
    for spk in dev_speakers:
        # Dev split: pick 1 utterance per speaker
        chosen = rng.choice(speaker_to_utts[spk])
        dev_entries.append(chosen)

    for spk in calib_speakers:
        chosen = rng.choice(speaker_to_utts[spk])
        calib_entries.append(chosen)

    for spk in test_speakers:
        chosen = rng.choice(speaker_to_utts[spk])
        test_entries.append(chosen)

    # Sort deterministically by utterance ID
    dev_entries.sort(key=lambda x: x["utt_id"])
    calib_entries.sort(key=lambda x: x["utt_id"])
    test_entries.sort(key=lambda x: x["utt_id"])

    return {
        "development": dev_entries,
        "calibration": calib_entries,
        "test": test_entries,
    }


def validate_splits(splits: Dict[str, List[Dict[str, Any]]]):
    """
    Asserts all 5 Task 3 validation requirements.
    """
    print("Validating split assertions...")

    dev_spks = set(e["speaker_id"] for e in splits["development"])
    calib_spks = [e["speaker_id"] for e in splits["calibration"]]
    test_spks = [e["speaker_id"] for e in splits["test"]]

    calib_spk_set = set(calib_spks)
    test_spk_set = set(test_spks)

    # Check 1: No speaker appears in multiple splits
    assert dev_spks.isdisjoint(calib_spk_set), "Speaker leakage between development and calibration!"
    assert dev_spks.isdisjoint(test_spk_set), "Speaker leakage between development and test!"
    assert calib_spk_set.isdisjoint(test_spk_set), "Speaker leakage between calibration and test!"
    print("  [Pass] Check 1: Strict speaker disjointness confirmed across all splits.")

    # Check 2: No speaker appears twice in calibration
    assert len(calib_spks) == len(calib_spk_set), "Duplicate speaker in calibration split!"
    print("  [Pass] Check 2: Exactly one utterance per speaker in calibration.")

    # Check 3: No speaker appears twice in test
    assert len(test_spks) == len(test_spk_set), "Duplicate speaker in test split!"
    print("  [Pass] Check 3: Exactly one utterance per speaker in test.")

    # Check 4: Every manifest entry resolves to existing audio files
    all_entries = splits["development"] + splits["calibration"] + splits["test"]
    for e in all_entries:
        assert os.path.isfile(e["prompt_wav_path"]), f"Prompt audio not found: {e['prompt_wav_path']}"
        assert os.path.isfile(e["gt_wav_path"]), f"GT audio not found: {e['gt_wav_path']}"
    print(f"  [Pass] Check 4: All {len(all_entries)} audio filepaths verified on disk.")

    # Check 5: Injected SNR is within 15.0 +/- 0.1 dB on sample test
    sample_entry = splits["development"][0]
    wav, sr = torchaudio.load(sample_entry["prompt_wav_path"])
    clean_np = wav.squeeze().numpy()
    noisy_np, noise, injected_snr = add_gaussian_noise_15db(clean_np, seed=123, target_snr_db=15.0)
    assert abs(injected_snr - 15.0) <= 0.1, f"Injected SNR {injected_snr} out of tolerance!"
    print(f"  [Pass] Check 5: Injected SNR tolerance confirmed ({injected_snr:.4f} dB within 15.0 +/- 0.1 dB).")


def build_manifest_rows(entries: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """Builds full dual-condition (clean & noisy15) manifest entries."""
    rows = []
    for e in entries:
        utt_id = e["utt_id"]
        input_seed = derive_input_seed(BASE_SEED, utt_id)

        # 1. Clean condition
        rows.append({
            "split": split_name,
            "utt_id": utt_id,
            "speaker_id": e["speaker_id"],
            "condition": "clean",
            "base_seed": BASE_SEED,
            "input_seed": input_seed,
            "prompt_text": e["prompt_text"],
            "prompt_wav_path": e["prompt_wav_path"],
            "gt_text": e["gt_text"],
            "gt_wav_path": e["gt_wav_path"],
        })

        # 2. Noisy15 condition
        rows.append({
            "split": split_name,
            "utt_id": utt_id,
            "speaker_id": e["speaker_id"],
            "condition": "noisy15",
            "base_seed": BASE_SEED,
            "input_seed": input_seed,  # Exact same seed per Task 3.3
            "prompt_text": e["prompt_text"],
            "prompt_wav_path": e["prompt_wav_path"],
            "gt_text": e["gt_text"],
            "gt_wav_path": e["gt_wav_path"],
            "target_snr_db": 15.0,
        })
    return rows


def main():
    print("=" * 80)
    print("TASK 3: GENERATING SPEAKER-DISJOINT DATA SPLITS")
    print("=" * 80)

    os.makedirs(SPLITS_DIR, exist_ok=True)

    print(f"Reading dataset: {META_LST}")
    all_entries = load_dataset_metadata(META_LST)
    unique_speakers = set(e["speaker_id"] for e in all_entries)

    print(f"Total dataset utterances: {len(all_entries)}")
    print(f"Unique prompt speakers:   {len(unique_speakers)}")

    splits = create_splits(
        all_entries,
        dev_n=DEV_SIZE,
        calib_n=CALIB_SIZE,
        test_n=TEST_SIZE,
        seed=SPLIT_SEED,
    )

    validate_splits(splits)

    # Generate condition rows and save JSON manifests
    summary = {
        "source_dataset": "Seed-TTS test-en (data/seed-tts-eval/en/meta.lst)",
        "split_seed": SPLIT_SEED,
        "base_seed": BASE_SEED,
        "total_source_utterances": len(all_entries),
        "total_source_speakers": len(unique_speakers),
        "splits": {
            "development": {
                "speakers": len(splits["development"]),
                "utterances": len(splits["development"]),
                "evaluations_with_conditions": len(splits["development"]) * 2,
            },
            "calibration": {
                "speakers": len(splits["calibration"]),
                "utterances": len(splits["calibration"]),
                "evaluations_with_conditions": len(splits["calibration"]) * 2,
            },
            "test": {
                "speakers": len(splits["test"]),
                "utterances": len(splits["test"]),
                "evaluations_with_conditions": len(splits["test"]) * 2,
            },
        },
        "conditions": ["clean", "noisy15"],
        "noise_specification": {
            "type": "Additive White Gaussian Noise (AWGN)",
            "snr_target_db": 15.0,
            "tolerance_db": 0.1,
            "clipping_policy": "np.clip(noisy, -1.0, 1.0)",
            "timing": "Applied directly to prompt audio before feature extraction and RMS normalization",
        },
    }

    for split_name in ("development", "calibration", "test"):
        manifest_rows = build_manifest_rows(splits[split_name], split_name)
        out_file = os.path.join(SPLITS_DIR, f"{split_name}_manifest.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest_rows, f, indent=2)
        print(f"Saved {len(manifest_rows)} rows to: {out_file}")

    summary_file = os.path.join(SPLITS_DIR, "split_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")

    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Development split:  {summary['splits']['development']['speakers']} speakers, {summary['splits']['development']['evaluations_with_conditions']} conditions")
    print(f"Calibration split:  {summary['splits']['calibration']['speakers']} speakers, {summary['splits']['calibration']['evaluations_with_conditions']} conditions")
    print(f"Test split:         {summary['splits']['test']['speakers']} speakers, {summary['splits']['test']['evaluations_with_conditions']} conditions")
    print("All checks passed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
