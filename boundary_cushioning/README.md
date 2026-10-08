# DMOSpeech 2 handover study

This folder turns the thesis problem statement into a controlled experiment. The [protocol](PROTOCOL.md) defines the hypotheses and decision rules. The pilot's 96 completed pairs are development evidence only; this folder uses a new set of 100 cross-sentence pairs from 20 speakers absent from the pilot.

## Current status

- `data/holdout_manifest.json` contains 100 selected pairs, five per speaker. All 100 reference recordings were checked: they exist, last 2.435–3.98 seconds, and have no speaker overlap with the pilot.
- Model-free preflight passed for all 100 pairs and all three planned random seeds.
- No held-out speech has been generated. No listener ratings have been collected. The study cannot yet establish an audible baseline defect or an improvement from the adaptive rule.

## Run on a machine with enough memory

From the repository root, use the project's Python environment. On Windows, the following commands use `venv\Scripts\python.exe`; replace that path with your environment's Python on another system.

```powershell
venv\Scripts\python.exe THESIS_EXPERIMENT\run_controlled.py --manifest THESIS_EXPERIMENT\data\holdout_manifest.json --output-dir THESIS_EXPERIMENT\results\holdout_grid8 --preflight
```

The preflight checks paths and configuration without loading the synthesis models. Then run the frozen primary configuration:

```powershell
venv\Scripts\python.exe THESIS_EXPERIMENT\run_controlled.py --manifest THESIS_EXPERIMENT\data\holdout_manifest.json --output-dir THESIS_EXPERIMENT\results\holdout_grid8 --seeds 1001 2002 3003 --teacher-grid 8 --device cuda
```

Each prompt and seed produces `baseline`, `baseline_copy`, `fixed`, and `proposed` audio from one shared teacher estimate, first student estimate, and later noise. The duplicate baseline must be byte identical to baseline; the runner stops if the waveforms differ. It writes per-case JSON and `metrics.csv` after each completed case, and resumes completed cases on rerun. Keep the `run_config.json` with the results. Run the app-default teacher grid of 16 in a **separate** output directory if resources permit.

The `--asr` option scores Whisper-base word error rate during generation. If memory is tight, generate audio first and score it in a separate pass with `score_asr.py` after generation. Speaker similarity remains blank unless a validated evaluator is added.

```powershell
venv\Scripts\python.exe THESIS_EXPERIMENT\score_asr.py --output-dir THESIS_EXPERIMENT\results\holdout_grid8 --preflight
venv\Scripts\python.exe THESIS_EXPERIMENT\score_asr.py --output-dir THESIS_EXPERIMENT\results\holdout_grid8
```

The separate scorer resumes by skipping WER values already present. Its preflight does not load Whisper.

## Analyze results and prepare listening

`analyze_results.py` pairs outputs by prompt and seed, checks the duplicate baseline, reports speaker-clustered intervals, and prepares blinded listener tasks. See [ANALYSIS.md](ANALYSIS.md) for commands and rating instructions. Freeze the listening subset and randomization seed before looking at held-out outcomes. Share only the packet's `listener` directory with raters; keep its answer keys private until their ratings are complete.

The primary claim requires independent human ratings. Objective audio metrics, including spectral flux, cannot substitute for those ratings.
