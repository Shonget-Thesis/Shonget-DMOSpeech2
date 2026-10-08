# Thesis Problem Statement: Evaluating Teacher-to-Student Handover in DMOSpeech 2

## What the pilot shows

The benchmark selected 100 cross-sentence speech pairs, and the corrected controlled comparison contains **96 completed pairs**. For the tested six-step teacher-guided setting, the baseline recorded a mean peak spectral-flux value of **60.33** and a mean utterance-level Whisper word error rate of **11.49%**. Changing how the first student estimate was accepted changed the generated results, but the effects varied across utterances. Adaptive TAGS recorded a mean peak flux of **59.92**; a matched fixed blend recorded **60.03**. The reported adaptive-versus-fixed difference is small, and its confidence interval includes zero. These results do not establish that the baseline has an audible handover defect or that adaptation is better than a simple fixed rule. See the [corrected five-way benchmark report](BENCHMARK_100_SAMPLES/COMPLETE_5WAY_100_SAMPLE_REPORT.md) and [saved paired results](BENCHMARK_100_SAMPLES/benchmark_5way_metrics.json).

## Problem statement

In baseline DMOSpeech 2 teacher-guided sampling, the teacher produces an estimated final mel representation, **T**. The student uses **T** to produce a first refined estimate, **S**, and the baseline accepts the full correction **S - T** before later student steps. The effect of that fixed acceptance decision on speech quality has not been established. The pilot shows that this intervention point can affect the output, but it does not identify when accepting the full correction helps or harms speech. The research problem is to determine whether teacher-to-student disagreement predicts meaningful quality loss and whether an inference-time acceptance rule can improve perceptual quality while preserving intelligibility and computation time.

## Research questions

1. When, if ever, is the first student correction **S - T** associated with poorer intelligibility, speaker similarity, naturalness, or audible artifacts?
2. Can a handover rule using information available during inference outperform both the original full-acceptance rule and a matched fixed blend on held-out speech?
3. What quality and latency tradeoffs does the rule introduce?

## Interpretation of the current measurements

The spectral-flux calculation covers the first 15% of the **generated waveform**. It does not directly measure the internal model handover, and a lower value alone does not establish better speech. The baseline performs endpoint reconstruction before one waveform decode; it does not splice teacher and student audio or implement one continuous velocity field. Claims of a boundary shock or mathematical continuity therefore need direct evidence. See the [baseline sampler](src/infer.py), [controlled benchmark](BENCHMARK_100_SAMPLES/run_fixed_w_extension.py), and [audio metric](BENCHMARK_SUITE/scripts/evaluate_audio.py).

## How this will be tested

The [controlled study protocol](THESIS_EXPERIMENT/PROTOCOL.md) defines a new, speaker-disjoint 100-pair test set, three fixed random seeds, and matched branches for the original full correction, a fixed blend, and the adaptive rule. The same teacher estimate, first student estimate, and later noise are reused within each case. A duplicate baseline branch must produce identical audio. Blinded listener preference is the primary outcome; word error rate, speaker similarity where a valid evaluator is available, runtime, and waveform diagnostics are secondary. The adaptive rule can be called a perceptual improvement only if held-out listener ratings support it against both the original baseline and the fixed blend. No such held-out results or listener ratings have been collected yet.
