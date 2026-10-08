# Paired evaluation and listening packet

`analyze_results.py` uses the Python standard library. It reads the new experiment's `results/holdout_grid8/metrics.csv` as one row per `(sample_id, speaker_id, seed, method)`. Required columns are `sample_id,speaker_id,seed,method`. The methods are `baseline`, `baseline_copy`, `fixed`, and `proposed` by default. Optional columns are:

| Column | Meaning |
|---|---|
| `wer` | Utterance word error rate as a fraction, such as `0.125` |
| `speaker_similarity` | Speaker embedding cosine similarity; larger is better |
| `inference_seconds` | Timed synthesis duration |
| `audio_seconds` | Generated audio duration, used with `inference_seconds` for real-time factor |
| `peak_flux`, `mean_flux` | Waveform spectral-flux diagnostics; these do not measure perceptual quality |
| `effective_w` | Accepted fraction of the first student correction |
| `disagreement` | Normalized handover ratio, `||S−T||/(||T||+epsilon)` |
| `audio_file` | Absolute path, or path relative to the directory containing `metrics.csv` |

## Analyze objective results

```powershell
python THESIS_EXPERIMENT/analyze_results.py analyze --results THESIS_EXPERIMENT/results/holdout_grid8/metrics.csv --manifest THESIS_EXPERIMENT/data/holdout_manifest.json --seeds 1001 2002 3003 --out THESIS_EXPERIMENT/results/holdout_grid8/analysis.md
```

The report pairs methods on the same sample and seed. It compares the proposed rule with both baseline and fixed blend. It reports equal-speaker-weighted differences and 95% confidence intervals from speaker bootstrap resampling. Missing metric pairs are excluded and counted separately. The duplicate baseline check compares audio files byte for byte when both are present. Runtime is excluded from the equality check because two executions can take different amounts of time.

For the older pilot JSON, use `--proposed adaptive_tanh --fixed fixed_w --results BENCHMARK_100_SAMPLES/benchmark_5way_metrics.json`. The pilot has no blinded ratings and is not a held-out confirmation test.

## Make a randomized A/B packet

```powershell
python THESIS_EXPERIMENT/analyze_results.py packet --results THESIS_EXPERIMENT/results/holdout_grid8/metrics.csv --out-dir THESIS_EXPERIMENT/listening_packet --max-cases 40 --naturalness-seed 1001 --naturalness-max-cases 100 --seed 2026
```

The command above requires exactly 40 A/B cases and 100 baseline naturalness cases. It stops if fewer are available. A/B cases are selected by seeded round-robin across speakers while spreading each speaker's cases across synthesis seeds, before listening results are known. The exact selected cases are recorded in `analyst/selection_manifest.csv`. The single-clip baseline task separately uses complete cases at one synthesis seed, ordinarily one clip for each distinct prompt. Use `--naturalness-seed 2002` to choose a different fixed synthesis seed, or `--naturalness-max-cases N` for a smaller speaker-balanced subset. Its cases are recorded in `analyst/naturalness_selection_manifest.csv`. Decide both subset sizes and seeds before listening; do not choose cases after inspecting model outcomes.

The command requires audio for every selected baseline, duplicate baseline, fixed, and proposed A/B case. It checks that each selected duplicate baseline WAV is byte-identical to its baseline WAV and stops if any differ. It refuses to overwrite an existing packet. It creates two A/B comparisons per selected case: proposed versus baseline, and proposed versus fixed. It also creates one single-clip baseline naturalness task per case at the chosen synthesis seed. Both task orders and A/B order are randomized using the given packet seed. The `listener` directory contains clips with only trial labels, `trials.csv`, `ratings_template.csv`, `naturalness_trials.csv`, and `naturalness_ratings_template.csv`. **Share only the `listener` directory with raters.** The separate `analyst/answer_key.csv` and `analyst/naturalness_answer_key.csv` contain method identities and must stay private until all ratings are collected.

Collect at least **three independent listener ratings per A/B pair and single clip**. Ask each listener to fill `listener_id` and `choice` (`A`, `B`, or `tie`) in a copy of `ratings_template.csv`. For `naturalness_ratings_template.csv`, fill `listener_id` and an integer `rating` from 1 (very unnatural) to 5 (very natural). No reference speech is provided, so the single-clip task measures naturalness rather than speaker fidelity. Combine completed pairwise rows into one CSV with `listener_id,trial_id,choice`, and single-clip rows into another CSV with `listener_id,trial_id,rating`. Then run:

```powershell
python THESIS_EXPERIMENT/analyze_results.py analyze --results THESIS_EXPERIMENT/results/holdout_grid8/metrics.csv --manifest THESIS_EXPERIMENT/data/holdout_manifest.json --seeds 1001 2002 3003 --ratings THESIS_EXPERIMENT/combined_ratings.csv --key THESIS_EXPERIMENT/listening_packet/analyst/answer_key.csv --naturalness-ratings THESIS_EXPERIMENT/combined_naturalness.csv --naturalness-key THESIS_EXPERIMENT/listening_packet/analyst/naturalness_answer_key.csv --out THESIS_EXPERIMENT/results/holdout_grid8/analysis_with_listening.md
```

The listening estimate gives each rated case equal weight within a speaker, then each speaker equal weight. Ties count as `0.5` preference for the proposed rule. The baseline naturalness score uses the same speaker weighting. The report tests whether normalized disagreement tracks baseline naturalness or preference for proposed using within-speaker Spearman associations only for speakers with at least four distinct rated prompts; these are exploratory associations, not causal proof. It lists naturalness coverage for each speaker. Confidence intervals resample speakers; they do not account for every source of listener variation. The report explicitly withholds any proposed-method perceptual-quality conclusion when pairwise ratings are absent. A listener preference result on a previously used tuning set is still exploratory.
