"""
SLR vs Baseline Discrepancy Diagnostic
Checks if per-sample SLR and Baseline WAVs are truly identical when w_slr=1.0
"""
import os, sys, json
import numpy as np
import soundfile as sf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(BENCHMARK_DIR, "results")
OUTPUTS_DIR   = os.path.join(BENCHMARK_DIR, "outputs")
METRICS_PATH  = os.path.join(RESULTS_DIR, "benchmark_100_metrics.json")

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples loaded: {len(data)}\n")
print(f"{'SID':>4} | {'rms_disp':>8} | {'w_slr':>6} | {'base_peak':>9} | {'slr_peak':>8} | {'dPeak':>7} | wav_match | max_sample_diff")
print("-" * 90)

mismatches = 0
for s in data:
    sid     = s["sample_id"]
    rms_r   = s["rms_displacement"]
    w_slr   = s["evaluations"]["slr_cap"]["effective_w"]
    b_peak  = s["evaluations"]["baseline"]["peak_flux"]
    sl_peak = s["evaluations"]["slr_cap"]["peak_flux"]
    delta   = sl_peak - b_peak

    # Load WAV files
    b_wav_path  = os.path.join(OUTPUTS_DIR, "baseline",  f"sample_{sid}_baseline.wav")
    sl_wav_path = os.path.join(OUTPUTS_DIR, "slr_cap",   f"sample_{sid}_slr_cap.wav")

    if not os.path.exists(b_wav_path) or not os.path.exists(sl_wav_path):
        print(f"{sid:>4} | {rms_r:>8.4f} | {w_slr:>6.4f} | {b_peak:>9.2f} | {sl_peak:>8.2f} | {delta:>+7.2f} | WAV_MISSING")
        continue

    b_wav,  sr_b  = sf.read(b_wav_path)
    sl_wav, sr_sl = sf.read(sl_wav_path)

    # Check shape
    same_shape = (b_wav.shape == sl_wav.shape)
    max_diff   = float(np.max(np.abs(b_wav - sl_wav))) if same_shape else float("inf")
    match      = "✓ SAME" if max_diff == 0.0 else f"✗ DIFF"

    if max_diff != 0.0:
        mismatches += 1

    if abs(delta) > 0.1 or max_diff > 0.0:
        print(f"{sid:>4} | {rms_r:>8.4f} | {w_slr:>6.4f} | {b_peak:>9.2f} | {sl_peak:>8.2f} | {delta:>+7.2f} | {match:10s} | {max_diff:.2e}")

print(f"\n{'='*90}")
print(f"Total WAV mismatches (w_slr=1.0 but wavs differ): {mismatches}/{len(data)}")

# Aggregate comparison
base_peaks = np.array([s["evaluations"]["baseline"]["peak_flux"] for s in data])
slr_peaks  = np.array([s["evaluations"]["slr_cap"]["peak_flux"]  for s in data])
diff_arr   = slr_peaks - base_peaks

print(f"\nAggregate peak flux difference (SLR - Baseline):")
print(f"  Mean  : {np.mean(diff_arr):+.4f}")
print(f"  Std   : {np.std(diff_arr):.4f}")
print(f"  Min   : {np.min(diff_arr):+.4f}")
print(f"  Max   : {np.max(diff_arr):+.4f}")
print(f"  # SLR > Base : {np.sum(diff_arr > 0)}")
print(f"  # SLR < Base : {np.sum(diff_arr < 0)}")
print(f"  # SLR == Base: {np.sum(diff_arr == 0.0)}")
