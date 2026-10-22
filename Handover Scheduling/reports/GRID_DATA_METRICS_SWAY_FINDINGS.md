# Findings: Grid Check, Seed-TTS Counts, Evaluation Measures, Sway-Grid Plot

**Date:** 2026-10-23
**Scope:** Four TO-DO items in the proposal *"A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching TTS Inference"*:

| # | TO-DO | Proposal location |
|---|---|---|
| 1 | Grid check: how "14 teacher steps at t_switch = 0.25" arises | Remark 3.1, §2.3.1 |
| 2 | Seed-TTS test-en utterance and speaker counts; split feasibility | §3.3.3, Table 3.4, Table 3.7 |
| 3 | Evaluation section (WER, SIM, UTMOS, RTF) | §2.6 (Eq. 2.23–2.25), §3.3.4, Table 3.5 |
| 4 | Sway-grid plot | §2.1.2, Figure 2.2 |

Every number below was recomputed for this report from:
- the released sampler, [`src/infer.py`](../../src/infer.py);
- the split manifests in [`data/splits/`](../data/splits/);
- the GPU candidate grid, [`colab/results/gpu_grid_all1000.csv`](../colab/results/gpu_grid_all1000.csv) (4,000 rows, 500 speakers);
- the Tesla T4 full-pipeline timing run, [`colab/results/gpu_full_pipeline_timing_3000.csv`](../colab/results/gpu_full_pipeline_timing_3000.csv) (3,000 rows).

It replaces the earlier [`PROPOSAL_TODO_CHECKS_REPORT.md`](PROPOSAL_TODO_CHECKS_REPORT.md). Section 5 lists where the two disagree.

---

## Summary

| Item | Finding | Proposal action |
|---|---|---|
| 14 teacher steps | Comes from a **32-interval** sway grid with a "last t_k ≤ 0.25" stop. That point is t₁₄ = **0.2270**, not 0.236 | Correct Remark 3.1 to t₁₄ ≈ 0.227 |
| Dataset size | Seed-TTS test-en: **1,088 utterances, 666 prompt speakers** | Fill in the TO DO in §3.3.3 |
| Splits | **100 / 200 / 200** speakers, zero overlap, one utterance per speaker. 500 of 666 used, 166 spare | Fill in the Table 3.4 sizes |
| Feasibility | n = 200 at α = 0.05, δ = 0.10 certifies with up to **5** added failures. Table 3.7 recomputed: **all 8 rows match** | No change |
| SIM model | Code loads **`microsoft/wavlm-base-plus-sv`** (WavLM Base+), while the proposal says WavLM-large | Name the model actually used, or re-score |
| RTF | The `rtf` column in the grid CSV excludes the teacher pass. Only the T4 timing run gives valid RTFs | Cite RTF only from the timing run, labelled as T4 |
| Sway plot | Two-panel figure regenerated from the formula: [`figures/sway_grid_tswitch.png`](../figures/sway_grid_tswitch.png) | Use it as Figure 2.2 |

---

## 1. Grid check: where "14 teacher steps at t_switch = 0.25" comes from

### 1.1 What the released code does

[`src/infer.py:520-523`](../../src/infer.py#L520-L523):

```python
t = torch.linspace(0, 1, teacher_steps + 1)                          # u_i = i / N
t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)   # s = -1  ->  t_i = 1 - cos(pi/2 * u_i)
t = t[:(t > teacher_stopping_time).float().argmax() + 2]
t = t[:-1]                                                           # keeps t_0 .. t_k, k = first index with t_k > s
```

- The code runs k Euler steps from t₀ to t_k. It then makes one more guided evaluation at t_k to get the endpoint estimate (lines 529–532, Eq. 3.3).
- So the code's rule is "stop at the first grid point above s". This matches Proposition 3.1: C(s) = 2(k + 1) + 3 with j = 1.

### 1.2 Grid points t_i = 1 − cos(π/2 · i/N)

| i (N = 16) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** |
|---|---|---|---|---|---|---|---|---|---|
| t_i | 0 | 0.0048 | 0.0192 | 0.0431 | 0.0761 | 0.1181 | 0.1685 | 0.2270 | **0.2929** |

| i (N = 32) | … | 12 | 13 | **14** | **15** | 16 |
|---|---|---|---|---|---|---|
| t_i | … | 0.1685 | 0.1968 | **0.2270** | **0.2590** | 0.2929 |

Every even point of the 32-interval grid is also a point of the 16-interval grid: t₂ᵢ⁽³²⁾ = tᵢ⁽¹⁶⁾.

### 1.3 Teacher steps at s = 0.25, by grid and stopping rule

| Grid | Stopping rule | k | t_k | Teacher Euler steps |
|---|---|---|---|---|
| 16 intervals (released default, **this study**) | first t_k > s (code) | 8 | 0.2929 | 8 (our reference Step 8, 21 NFE) |
| 16 intervals | last t_k ≤ s | 7 | 0.2270 | 7 |
| 32 intervals | first t_k > s (code) | 15 | 0.2590 | 15 |
| **32 intervals** | **last t_k ≤ s** | **14** | **0.2270** | **14 ← the paper's number** |

**Conclusions**
- Exactly 14 teacher steps at s = 0.25 happens only on a **32-interval** grid read as "last point not above s". The same count also results from the code rule with s anywhere in [0.1968, 0.2270).
- The released `generate_teacher_only` defaults to `teacher_steps=32` ([`src/infer.py:341`](../../src/infer.py#L341)). This supports the 32-interval reading.
- **Remark 3.1 has one wrong number.** On the 32-interval grid, t₁₄ = 0.2270, not ≈ 0.236. A t₁₄ of 0.236 would need N ≈ 31.4, which is not an integer (N = 31 gives 0.2412).
- The paper's handover time (0.2270) is the same flow time as our **Step 7**. The paper reaches it with 14 finer steps, where our grid takes 7.

### 1.4 Forward-pass cost of the paper's configuration

| Count | Teacher passes | Student passes | Total | Full teacher (N = 32) | Ratio |
|---|---|---|---|---|---|
| Paper, no endpoint evaluation | 2 × 14 = 28 | 2 | 30 | 64 | 2.13× |
| Released code, with endpoint evaluation | 2 × 15 = 30 | 2 | 32 | 64 | 2.00× |

- The paper's 2 student steps mean starting at student index j = 2. The student grid is {0, 0.0761, 0.2929, 0.6173}, so the remaining points are τ ∈ {0.2929, 0.6173}. This study fixes j = 1, which gives 3 student passes.
- The paper's reported ~1.8× wall-clock speed-up is below both pass ratios. That is expected, because the vocoder, the duration model and data movement do not shrink with NFE.

**Still open:** the authors' exact evaluation script is not public. Phrase Remark 3.1 as "consistent with the released code (default `teacher_steps=32`)", not as confirmed.

---

## 2. Seed-TTS test-en counts and split feasibility

### 2.1 Corpus counts

From [`data/splits/split_summary.json`](../data/splits/split_summary.json), written by [`data/make_splits.py`](../data/make_splits.py) from `seed-tts-eval/en/meta.lst`. `meta.lst` itself is not in this checkout, so the two corpus totals rely on the run that built the splits.

| Quantity | Value |
|---|---|
| Utterances in Seed-TTS test-en | **1,088** |
| Unique prompt speakers | **666** (≈ 1.63 utterances per speaker) |
| Prompt source | Common Voice (Anastassiou et al., 2024; Ardila et al., 2020) |
| Target-text length (500 utterances used) | 3–21 words; median 11; mean 10.96; 10th–90th percentile 8–14; **8.4 %** below 8 words |

### 2.2 Splits (for Table 3.4)

These counts were recomputed from the three manifests:

| Split | Speakers | Utterances | Rows (× clean, noisy15) | Used for |
|---|---|---|---|---|
| Development | 100 | 100 | 200 | Objective 1; choice of SNR gate |
| Calibration | 200 | 200 (1 per speaker) | 400 | Objective 3, threshold λ̂ |
| Test | 200 | 200 (1 per speaker) | 400 | Objective 3 held-out check; Objective 4 |
| Not used | 166 | — | — | Reserve |

- Speaker overlap between splits is **0** for all three pairs.
- The split seed is 42.
- `make_splits.py` also asserts that every audio path exists and that the injected noise SNR is within 15.0 ± 0.1 dB.

### 2.3 Feasibility: can 200 calibration inputs certify α = 5 %?

For each row, the table gives the largest number of added failures k with P(Bin(n, α) ≤ k) ≤ δ. This is the exact binomial check of Eq. 3.12. **All rows match the proposal's Table 3.7.**

| n | α | δ | Max. added failures |
|---|---|---|---|
| 45 | 0.05 | 0.10 | 0 (minimum n = ⌈ln 0.10 / ln 0.95⌉ = 45) |
| 50 | 0.05 | 0.05 | none certifiable |
| 50 | 0.10 | 0.10 | 1 |
| 100 | 0.05 | 0.10 | 1 |
| 100 | 0.10 | 0.10 | 5 |
| **200** | **0.05** | **0.10** | **5** |
| 200 | 0.10 | 0.10 | 14 |
| 300 | 0.05 | 0.10 | 9 |

- The Hoeffding alternative would need n ≥ ln(1/δ)/(2α²) = 461 inputs. That is more than the 200 available per split, which is why the exact binomial is used. This confirms the figure of 461 in §2.5.
- **Verdict: feasible.** The splits need 500 of the 666 speakers. n = 200 tolerates up to 5 added failures (an observed rate of 2.5 %).
- **Limit on the held-out check:** a single split cannot separate 2.5 % from 5 %. With 5 failures in 200, the 95 % Clopper-Pearson interval is 0.82 %–5.74 %. The repeated-split check (`phase2_calib_ltt_200spk/stability.py`) therefore matters.

### 2.4 Caveat: a "speaker" is a prompt clip

- `make_splits.py` takes the speaker ID from the prompt file name (`common_voice_en_<clip>`). That is a Common Voice **clip ID**, not the contributor (`client_id`).
- Two clips from the same real person could therefore land in different splits.
- Heuristic check: in this corpus, the prompt and target clip IDs of one utterance are close together (median gap 2). Across all 1,000 clip IDs in the three splits, **no two clips from different splits are within 3 IDs of each other**. This shows no sign of leakage, but it is not proof.
- Suggested wording: *"speaker-disjoint at the level of Seed-TTS prompt identities."*

---

## 3. Evaluation section: WER, SIM, UTMOS, RTF

### 3.1 Draft text for §2.6 / §3.3.4

**Intelligibility (WER).** Each output is transcribed with Whisper-large-v3 (Radford et al., 2023), and the transcript is compared with the target text:

$$\mathrm{WER} = \frac{S + D + I}{N} \qquad \text{(Equation 2.23)}$$

S, D and I are the substitutions, deletions and insertions in the minimum edit alignment between the normalized reference and the hypothesis. N is the number of reference words. Because insertions are not bounded by N, WER can exceed 100 %; looping or hallucinated outputs on noisy prompts reach 3,330 % in our data. Before scoring, both texts are normalized as in the Seed-TTS / F5-TTS English evaluation: punctuation is replaced by spaces, whitespace is collapsed and text is lowercased.

A per-utterance WER on short sentences is coarse. Target sentences have a median of 11 words, so one wrong word costs about 9 points. For the 8.4 % of sentences with fewer than 8 words, one wrong word already exceeds 12.5 %. The ceiling c = 12.5 % in Definition 3.1 equals one error in an 8-word sentence. It therefore lets most sentences (8 words or more, 91.6 %) absorb one misrecognized word without counting it as a failure.

**Speaker similarity (SIM).** SIM is the cosine similarity between speaker embeddings of the generated speech and of the ground-truth target recording, as in DMOSpeech 2:

$$\mathrm{SIM} = \frac{e_{\text{gen}}^{\top} e_{\text{ref}}}{\lVert e_{\text{gen}} \rVert\, \lVert e_{\text{ref}} \rVert} \qquad \text{(Equation 2.24)}$$

e_gen and e_ref are the verifier's embeddings of the generated audio and of the original target recording. SIM depends on the verifier, so values are comparable only when the same verifier produced them. The failure rule therefore compares each candidate only with the reference configuration, scored by the same verifier on the same input.

**Naturalness (UTMOS).** UTMOS (Saeki et al., 2022) predicts the mean opinion score listeners would give. It was built for the VoiceMOS Challenge. It is reliable for ranking systems on average but noisy for single utterances. It is therefore reported but not used in the failure rule, and it does not replace a listening test.

**Cost (NFE and RTF).** The primary cost is the number of network forward passes, with a guided teacher step counting as two. It follows exactly from Proposition 3.1 and does not depend on hardware. The secondary cost is the real-time factor:

$$\mathrm{RTF} = \frac{T_{\text{synthesis}}}{T_{\text{audio}}} \qquad \text{(Equation 2.25)}$$

T_synthesis is the wall-clock time from prepared inputs to the decoded waveform (teacher, student and vocoder). T_audio is the duration of the generated speech. RTF < 1 means faster than real time. RTF includes costs that NFE does not count, such as the vocoder, the duration model and data movement. Savings in RTF are therefore smaller than savings in NFE.

### 3.2 What the code actually uses ([`core/metrics.py`](../core/metrics.py))

| Measure | Implementation | Matches proposal? |
|---|---|---|
| WER | HF pipeline `openai/whisper-large-v3`, FP32, `language="english"`, resampled to 16 kHz. Scored with `jiwer` after `normalize_text` | ✅ |
| SIM | `microsoft/wavlm-base-plus-sv` ([`core/metrics.py:246`](../core/metrics.py#L246)), x-vector head, cosine similarity against the ground-truth target | ❌ **WavLM Base+, not WavLM-large** |
| UTMOS | `tarepan/SpeechMOS:v1.2.0`, `utmos22_strong` | ✅ |
| Failure rule | `check_failure`: WER > max(0.125, WER_ref + 0.01) **or** SIM < SIM_ref − 0.02, with constants from [`core/failure_config.json`](../core/failure_config.json) | ✅ |
| NFE | Proposition 3.1: Steps 4 / 5 / 6 / 8 → 13 / 15 / 17 / 21 | ✅ |

**SIM verifier mismatch**
- Table 3.5, §3.3.4 and the docstring of `WavLMSimilarityEvaluator` all say "WavLM-large". The model actually loaded is WavLM Base+ SV.
- Absolute SIM (≈ 0.938 on clean prompts) is therefore **not comparable** with published DMOSpeech 2 numbers, which use the WavLM-large Seed-TTS verifier.
- The failure rule and every internal comparison remain **valid**, because they compare each candidate with Step 8 under the same verifier.
- There are two ways to fix this:
  - **(a) Recommended:** state the actual model in Chapter 3 and drop any comparison with published SIM. No GPU time is needed.
  - **(b)** Re-score all audio with WavLM-large. This needs a GPU re-run, and failure labels may shift.
- In either case, fix the "WavLM-large" docstring in `core/metrics.py`.

**Ground-truth fallback:** if the target recording is missing, [`run_gpu_expanded_benchmark.py:348-353`](../colab/run_gpu_expanded_benchmark.py#L348-L353) silently scores SIM against the **prompt** audio instead. The grid CSV does not record which reference was used, so the number of affected rows cannot be checked from the results. Before submission, either log a flag for it or confirm that every `wavs/` file was present on Colab.

### 3.3 RTF: which numbers are valid

| Source | What is timed | Valid? |
|---|---|---|
| `rtf` column in `gpu_grid_all1000.csv` (and the phase 1–3 grids) | Only `run_student_completion`: student passes and vocoder. The shared teacher pass is excluded | ❌ ≈ 0.11 at **every** step. Do not cite |
| `gpu_full_pipeline_timing_3000.csv` (Tesla T4, one session) | Full pipeline per policy: teacher, student and vocoder | ✅ |

**Valid T4 RTF (mean).** The [`gpu_timing_summary_by_condition.csv`](../colab/results/gpu_timing_summary_by_condition.csv) file and the timing report use the **test split only**. Label it that way:

| Condition | Cohort | Step 8 | Proposed (SNR + drift) | SNR → Step 4 |
|---|---|---|---|---|
| Clean | Test split (200) | 0.725 | 0.529 | 0.453 |
| Clean | All 500 speakers | 0.711 | 0.518 | 0.442 |
| Noisy15 | Test split (200) | 0.706 | 0.706 | 0.706 |
| Noisy15 | All 500 speakers | 0.693 | 0.693 | 0.693 |

- On noisy prompts, all three policies fall back to Step 8 (21 NFE), so their RTFs are identical.
- On clean prompts across all 500 speakers, the proposed policy averages 15.29 NFE.
- §3.3.5 promises runtimes on the study's Ryzen 7 laptop. **None have been measured yet**, so every valid RTF is a T4 number.

### 3.4 Observed quality per step (GPU grid, 500 speakers, 4,000 rows)

| Condition | Step | NFE | Mean WER | Median WER | SIM | UTMOS (SD) | Added-failure rate |
|---|---|---|---|---|---|---|---|
| Clean | 4 | 13 | 1.60 % | 0 % | 0.938 | 3.86 (0.43) | 2.4 % |
| Clean | 5 | 15 | 1.59 % | 0 % | 0.938 | 3.87 (0.44) | 1.8 % |
| Clean | 6 | 17 | 1.70 % | 0 % | 0.938 | 3.87 (0.44) | 1.0 % |
| Clean | 8 (ref) | 21 | 1.65 % | 0 % | 0.938 | 3.87 (0.44) | — |
| Noisy15 | 4 | 13 | 75.0 % | 15.4 % | 0.694 | 1.25 (0.02) | 64.4 % |
| Noisy15 | 5 | 15 | 74.3 % | 10.0 % | 0.699 | 1.25 (0.04) | 53.8 % |
| Noisy15 | 6 | 17 | 29.6 % | 9.1 % | 0.705 | 1.25 (0.02) | 49.4 % |
| Noisy15 | 8 (ref) | 21 | 36.1 % | 10.0 % | 0.712 | 1.25 (0.02) | — |

Points worth a sentence in the thesis:
- **Clean prompts:** mean quality is flat across steps. The reference has WER = 0 on 86.0 % of utterances.
- **Noisy prompts are hard even for the reference:** Step 8 exceeds the 12.5 % WER ceiling on 43.6 % of them.
- **UTMOS hits its floor on noisy prompts** (≈ 1.25, SD ≈ 0.02). The model copies the prompt's noise into the output, so UTMOS carries no information there. This is another reason it stays out of the failure rule.
- **Noisy mean WER is driven by a few runaway transcripts** (maximum 3,330 %). The medians (15.4 / 10.0 / 9.1 / 10.0 %) are better behaved. This is why failures are counted per utterance rather than compared by means.
- **Failure causes** across the 3,000 non-reference rows: 232 WER-only, 453 SIM-only and 179 both. SIM is the more frequent trigger.

---

## 4. Sway-grid plot (Figure 2.2)

![Sway grid and grid check](../figures/sway_grid_tswitch.png)

- **File:** [`figures/sway_grid_tswitch.png`](../figures/sway_grid_tswitch.png), regenerated today by [`figures/generate_sway_grid_check.py`](../figures/generate_sway_grid_check.py). The script uses only the formula, no data.
- **Panel (a)** implements the proposal's figure spec (the TO-DO at §2.1.3):
  - u = i/16 on the x-axis and flow time t on the y-axis;
  - s = 0 shown dashed, s = −1 shown solid;
  - the 17 teacher grid points;
  - a horizontal line at t_switch = 0.25, with the teacher region below it shaded;
  - candidate Steps 4 / 5 / 6 / 8 marked with their t and NFE.
- **Annotations for the §2.1.2 TO-DO, point (a):**
  - **8 of the 17 grid points (t₀–t₇) lie at or below 0.25.**
  - The first half of the grid (u ≤ 0.5) covers only t ≤ 0.293. That is the "half of the grid points fall below t ≈ 0.29" statement.
- **Panel (b)** illustrates Section 1: the 16- and 32-interval grids near s = 0.25 and the stop index under each rule.
  - It is best placed beside Remark 3.1 in Chapter 3.
  - If Figure 2.2 should be a single panel, crop to panel (a).
- **Older figure:** [`figures/sway_grid_plot.png`](../figures/sway_grid_plot.png) plots against the step index, has no t_switch line, and its Step 4/5 labels overlap. Do not use it.

---

## 5. Corrections to the earlier `PROPOSAL_TODO_CHECKS_REPORT.md`

| Claim in earlier report | Recomputed | Note |
|---|---|---|
| 7.8 % of sentences have < 8 words (92.2 % ≥ 8) | **8.4 %** (91.6 % ≥ 8) | Over the 500 used utterances |
| Maximum noisy per-utterance WER 9,867 % | **3,330 %** | From `gpu_grid_all1000.csv` |
| RTF 0.725 / 0.529 / 0.453 given without a cohort label | Correct for the **test split only**. All 500 speakers: 0.711 / 0.518 / 0.442 | Label the cohort |
| Test-split CP interval "0.82–5.74 %" for a 2.5 % rate | Correct for **5 / 200** | Unchanged |
| The 166 unused speakers were "later used in Phase 12" | Not found in the repository's results | Dropped; listed only as a reserve |

All other numbers in the earlier report (grid values, stop indices, Table 3.7, split sizes, metric models, per-step quality, failure causes) were reproduced exactly.

---

## 6. Proposal edits

1. **Remark 3.1:** change "t₁₄ ≈ 0.236" to "t₁₄ ≈ 0.227". Note that this equals the 16-interval t₇. Cite `teacher_steps=32` in `generate_teacher_only` as support, and phrase the reading as "consistent with the released code".
2. **§2.3.1:** paper configuration = 30 forward passes vs. 64 (2.13×), or 32 vs. 64 with the endpoint evaluation. The paper's 2 student steps correspond to j = 2.
3. **§3.3.3:** "1,088 utterances from 666 prompt speakers". **Table 3.4** sizes: 100 / 200 / 200. Add the prompt-identity caveat.
4. **Table 3.5 / §3.3.4:** name WavLM-Base+ SV, or re-score with WavLM-large. Delete the "[TO DO: update the runners…]" note only after resolving SIM.
5. **§2.6:** insert the text from Section 3.1 (Eq. 2.23–2.25).
6. **Results:** cite RTF only from the full-pipeline timing run, labelled "Tesla T4" with its cohort. State that the Ryzen 7 runtime has not been measured.
7. **Figure 2.2:** use `sway_grid_tswitch.png`, panel (a) or both panels.
