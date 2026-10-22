# Proposal TO-DO Checks: Grid, Data, Evaluation Measures, Sway Figure

This report closes four TO-DO items from the proposal ("A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching TTS Inference"):

1. Grid check: how "14 teacher steps at t_switch = 0.25" arises (Remark 3.1)
2. Seed-TTS test-en utterance and speaker counts, and split feasibility (§3.3.3, Table 3.4)
3. The evaluation-measures section: WER, SIM, UTMOS, RTF (§2.6, §3.3.4, Table 3.5)
4. The sway-grid plot (Figure 2.2)

Every number below was computed from the repository: the released sampler (`src/infer.py`), the split files (`data/splits/`), the GPU grid (`colab/results/gpu_grid_all1000.csv`, 4,000 rows) and the T4 timing run (`colab/results/gpu_full_pipeline_timing_3000.csv`).

## Summary

| Item | Result | Proposal action |
|---|---|---|
| 14 teacher steps | Comes from a **32-interval** sway grid with a "last t_k ≤ 0.25" stop: t₁₄ = **0.2270** | Remark 3.1 says t₁₄ ≈ 0.236. **Correct it to 0.2270** |
| Dataset counts | **1,088 utterances, 666 prompt speakers** | Fill in the TO DO in §3.3.3 |
| Split feasibility | 100 / 200 / 200 is feasible: 500 of 666 speakers, 166 spare. Table 3.7 re-computed and **matches exactly** | Fill Table 3.4 sizes |
| SIM model | Code uses **`microsoft/wavlm-base-plus-sv`**, but the proposal and code comments say "WavLM-large" | **Mismatch: fix the text or re-run** (Section 3.2) |
| RTF in grid CSVs | The `rtf` column times only the student + vocoder; the teacher pass is excluded | **Do not cite it.** Use the T4 timing run |
| Sway figure | New two-panel figure, drawn from the formula | Replace Figure 2.2 |

---

## 1. Grid check: where "14 teacher steps at t_switch = 0.25" comes from

### How the released code stops

[`src/infer.py:520-523`](../../src/infer.py#L520-L523):

```python
t = torch.linspace(0, 1, teacher_steps + 1)                    # uniform u_i = i / N
t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)   # s = -1  ->  t = 1 - cos(pi/2 * u)
t = t[:(t > teacher_stopping_time).float().argmax() + 2]
t = t[:-1]                                                     # keeps t_0 ... t_k, k = first index with t_k > s
```

- It runs k Euler steps (t₀ → t_k) and then one more guided evaluation at t_k for the endpoint (lines 529-532).
- So the code's rule is "stop at the first grid point **above** s". This matches the proposal's Algorithm 1 and Proposition 3.1: C(s) = 2(k + 1) + 3.

### Sway grid points t_i = 1 − cos(π/2 · i/N)

| i (N = 16) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** |
|---|---|---|---|---|---|---|---|---|---|
| t_i | 0 | 0.0048 | 0.0192 | 0.0431 | 0.0761 | 0.1181 | 0.1685 | 0.2270 | **0.2929** |

| i (N = 32) | … | 12 | 13 | **14** | **15** | 16 |
|---|---|---|---|---|---|---|
| t_i | … | 0.1685 | 0.1968 | **0.2270** | **0.2590** | 0.2929 |

Every even point of the 32-interval grid is a point of the 16-interval grid (t₂ᵢ⁽³²⁾ = tᵢ⁽¹⁶⁾).

### Teacher steps at s = 0.25 under each grid and rule

| Grid | Stopping rule | Stop index k | Stop time t_k | Teacher Euler steps |
|---|---|---|---|---|
| 16 intervals (released code, **this study**) | first t_k > s (code) | 8 | 0.2929 | 8, i.e. our reference Step 8, 21 NFE |
| 32 intervals | first t_k > s (code) | 15 | 0.2590 | 15 |
| **32 intervals** | **last t_k ≤ s** | **14** | **0.2270** | **14 ← the paper's number** |

**Conclusion:**
- "14 teacher steps at t_switch = 0.25" arises only on a 32-interval grid with a "last point not above s" reading.
- Equivalently, it is the code rule with s anywhere in [0.1968, 0.2270).
- Supporting evidence: the released `generate_teacher_only` defaults to `teacher_steps=32` ([`src/infer.py:341`](../../src/infer.py#L341)), so the authors did evaluate on a 32-interval grid.
- The proposal's reading in Remark 3.1 is therefore right, **except for one number: t₁₄ = 0.2270, not ≈ 0.236.**
- The paper's handover point (t = 0.2270) is the same flow time as our **Step 7**, reached with 14 finer steps instead of 7.

### Forward-pass cost of the paper's configuration

| Count | Teacher | Student | Total | Full teacher | Speed-up in passes |
|---|---|---|---|---|---|
| Paper (no endpoint evaluation) | 2 × 14 = 28 | 2 | 30 | 2 × 32 = 64 | 2.13× |
| Released code (adds the endpoint evaluation) | 2 × (14 + 1) = 30 | 2 | 32 | 64 | 2.00× |

- The paper's 2 student steps correspond to start index j = 2 (τ ∈ {0.2929, 0.6173}). This study fixes j = 1 (3 student steps).
- The paper's reported 1.8× runtime speed-up is below either pass ratio. That is expected, because the vocoder, duration predictor and data movement do not shrink with NFE.

**Remaining TO DO:** "confirm the 32-interval reading against the authors' evaluation settings". The arithmetic and the `teacher_steps=32` default both support the reading, but the authors' exact evaluation script is not in the repo. Phrase it as "consistent with the released code".

---

## 2. Seed-TTS test-en counts and split feasibility

### Counts

Source: [`data/splits/split_summary.json`](../data/splits/split_summary.json), written by [`data/make_splits.py`](../data/make_splits.py) from `seed-tts-eval/en/meta.lst`. The `meta.lst` file itself is not in this checkout; the counts come from the run that built the splits.

| Quantity | Value |
|---|---|
| Utterances in Seed-TTS test-en | **1,088** |
| Unique prompt speakers | **666** (about 1.63 utterances per speaker) |
| Prompt source | Common Voice (Anastassiou et al., 2024; Ardila et al., 2020) |
| Target-text length (500 used utterances) | 3–21 words, median 11, mean 11.1 (10th–90th percentile: 8–14) |

### Splits (fill Table 3.4)

| Split | Speakers | Utterances | Evaluations (× clean, noisy15) | Used for |
|---|---|---|---|---|
| Development | 100 | 100 | 200 | Objective 1, SNR gate choice |
| Calibration | 200 | 200 (1 per speaker) | 400 | Objective 3, LTT threshold |
| Test | 200 | 200 (1 per speaker) | 400 | Objective 3 held-out check, Objective 4 |
| Unused | 166 | — | — | Later used as fresh speakers in Phase 12 |

Split seed 42. `make_splits.py` asserts:
- zero speaker overlap across splits;
- one utterance per speaker in calibration and test;
- every audio path exists;
- injected noise SNR within 15.0 ± 0.1 dB.

### Feasibility: can 200 calibration speakers certify α = 5%?

I re-computed Table 3.7 with the exact binomial (the largest number of failures k with P(Bin(n, α) ≤ k) ≤ δ). **It matches the proposal exactly:**

| n | α | δ | Max. failures that still certify |
|---|---|---|---|
| 45 | 0.05 | 0.10 | 0 (minimum n = ⌈ln 0.10 / ln 0.95⌉ = 45) |
| 50 | 0.05 | 0.05 | none certifiable |
| 50 | 0.10 | 0.10 | 1 |
| 100 | 0.05 | 0.10 | 1 |
| 100 | 0.10 | 0.10 | 5 |
| **200** | **0.05** | **0.10** | **5** |
| 200 | 0.10 | 0.10 | 14 |
| 300 | 0.05 | 0.10 | 9 |

- **Verdict: feasible.** 500 of 666 speakers are needed. n = 200 allows up to 5 added failures, which is enough room to certify a non-trivial threshold. In practice, λ̂ = 0.50 was certified.
- On the test split, a 2.5% rate has a Clopper-Pearson 95% interval of 0.82–5.74%. That's wide enough that a single split can't separate 2.5% from 5%. This is why the repeated-split check (`stability.py`) matters.

### Caveat: "speaker" means a prompt clip

- `make_splits.py` takes the speaker ID from the prompt file name (`common_voice_en_<clip>`). That is a Common Voice **clip** ID, not the contributor (`client_id`).
- So two prompt clips could, in principle, belong to the same real person and land in different splits.
- Heuristic check: prompt and target clip IDs of one utterance are usually consecutive (median gap 2). I found **no pair of clips from different splits within 3 IDs of each other**. So there's no sign of leakage, but without Common Voice contributor metadata it can't be proven.
- Suggested wording: "speaker-disjoint at the level of Seed-TTS prompt identities".

---

## 3. Evaluation section draft (WER, SIM, UTMOS, RTF)

Ready-to-adapt text for §2.6 / §3.3.4, followed by the implementation facts and two corrections.

### 3.1 Draft text

**Intelligibility (WER).** Each output is transcribed with Whisper-large-v3 (Radford et al., 2023), and the transcript is compared with the target text:

$$\mathrm{WER} = \frac{S + D + I}{N} \qquad \text{(Equation 2.23)}$$

- S, D and I are the substitutions, deletions and insertions in the minimum edit alignment between the normalized reference and the hypothesis.
- N is the number of reference words.
- WER can exceed 100% when insertions are many.
- Before scoring, both texts are normalized as in the Seed-TTS / F5-TTS English evaluation: punctuation is replaced by spaces, whitespace is collapsed, and text is lowercased.
- Per-utterance WER is coarse. The target sentences have a median of 11 words, so one wrong word costs about 9 points; for the 7.8% of sentences with fewer than 8 words, one wrong word already exceeds 12.5%.
- The absolute ceiling c = 12.5% in Definition 3.1 is set at one error in an 8-word sentence. It therefore lets the large majority of sentences (≥ 8 words, 92.2%) absorb a single misrecognized word without counting it as a failure.

**Speaker similarity (SIM).** SIM is the cosine similarity between speaker embeddings of the generated speech and of the ground-truth target recording:

$$\mathrm{SIM} = \frac{e_{\text{gen}}^{\top} e_{\text{ref}}}{\lVert e_{\text{gen}} \rVert\, \lVert e_{\text{ref}} \rVert} \qquad \text{(Equation 2.24)}$$

- e_gen and e_ref are the verifier's embeddings of the generated audio and of the original target recording (as in DMOSpeech 2).
- SIM depends on the verifier, so values are comparable only within one verifier.
- The failure rule therefore compares each candidate only with the reference configuration scored by the same verifier.

**Naturalness (UTMOS).** UTMOS (Saeki et al., 2022) predicts the mean opinion score listeners would give.
- It was built for the VoiceMOS Challenge. It is reliable for ranking systems on average but noisy for single utterances.
- It is therefore reported but not used in the failure rule. It does not replace a listening test.

**Cost (NFE and RTF).** The primary cost is the number of network forward passes, which follows exactly from Proposition 3.1 and does not depend on hardware. The secondary cost is the real-time factor:

$$\mathrm{RTF} = \frac{T_{\text{synthesis}}}{T_{\text{audio}}} \qquad \text{(Equation 2.25)}$$

- T_synthesis is the wall-clock time from the prepared inputs to the decoded waveform: teacher, student and vocoder.
- T_audio is the duration of the generated speech.
- RTF < 1 means faster than real time.
- RTF includes costs that NFE does not count (the vocoder, the duration model and data movement), so savings in RTF are smaller than savings in NFE.

### 3.2 What the code actually uses (`core/metrics.py`)

| Measure | Implementation | Matches the proposal? |
|---|---|---|
| WER | HF pipeline `openai/whisper-large-v3`, FP32, `language="english"`, resampled to 16 kHz; `jiwer` after `normalize_text` (punctuation → space, collapse whitespace, lowercase) | ✅ |
| SIM | `microsoft/wavlm-base-plus-sv` (WavLM **Base+** with an x-vector head), cosine vs. the ground-truth target recording | ❌ **Not WavLM-large** (see below) |
| UTMOS | `tarepan/SpeechMOS:v1.2.0`, `utmos22_strong` | ✅ |
| Failure rule | `check_failure`: WER > max(0.125, WER_ref + 0.01) or SIM < SIM_ref − 0.02, constants from `failure_config.json` | ✅ |
| NFE | Proposition 3.1, Steps 4/5/6/8 = 13/15/17/21 | ✅ |

**SIM verifier mismatch.**
- The proposal (Table 3.5), DMOSpeech 2 and the code's own comments say "WavLM-large-based verifier". The loaded model is WavLM Base+ SV.
- Consequence 1: our absolute SIM (≈ 0.94 on clean prompts) is **not comparable** with published DMOSpeech 2 SIM numbers, which use the WavLM-large Seed-TTS verifier and sit far lower.
- Consequence 2: the **failure rule and all internal comparisons stay valid**. They are relative: candidate vs. Step 8, same verifier.
- Two options:
  - (a) **State the model honestly** in Chapter 3 ("WavLM-Base+ speaker-verification model") and drop the comparison with published SIM. **Recommended:** no GPU needed.
  - (b) Re-score all audio with the WavLM-large verifier. This needs a full GPU re-run, and failure labels could shift.
- Either way, fix the docstrings in `core/metrics.py` and the "WavLM-large" print statements in the runners.

**Ground-truth fallback.** If a target recording is missing, `run_gpu_expanded_benchmark.py` silently scores SIM against the **prompt** instead. Report how many rows used the fallback, or state that none did.

### 3.3 RTF: which numbers are valid

| Source | What is timed | Valid RTF? |
|---|---|---|
| `rtf` column in the grid CSVs (`gpu_grid_all1000.csv`, phase 1–3 grids) | Only `run_student_completion` (3 student passes + vocoder). The shared teacher pass is excluded | ❌ ≈ 0.11 for **every** step, because the teacher cost is missing. **Do not cite** |
| `gpu_full_pipeline_timing_3000.csv` (3,000 timed runs, Tesla T4) | Full pipeline per policy (teacher + student + vocoder) | ✅ |

**Valid T4 RTF, by condition:**

| Condition | Step 8 | Proposed (SNR + drift) | SNR → Step 4 |
|---|---|---|---|
| Clean | 0.725 (3.29 s) | 0.529 (2.40 s) | 0.453 (2.06 s) |
| Noisy15 | 0.706 | 0.706 | 0.706 (all fall back to Step 8) |

§3.3.5 promises runtimes on the Ryzen 7 laptop. **None have been measured yet**; all valid RTFs are T4 numbers.

### 3.4 Observed quality per step (GPU grid, 500 speakers)

| Condition | Step | NFE | WER | SIM | UTMOS |
|---|---|---|---|---|---|
| Clean | 4 | 13 | 1.60% | 0.938 | 3.86 |
| Clean | 5 | 15 | 1.59% | 0.938 | 3.87 |
| Clean | 6 | 17 | 1.70% | 0.938 | 3.87 |
| Clean | 8 (ref) | 21 | 1.65% | 0.938 | 3.87 |
| Noisy15 | 4 | 13 | 75.0% | 0.694 | 1.25 |
| Noisy15 | 5 | 15 | 74.3% | 0.699 | 1.25 |
| Noisy15 | 6 | 17 | 29.6% | 0.705 | 1.25 |
| Noisy15 | 8 (ref) | 21 | 36.1% | 0.712 | 1.25 |

Observations worth a sentence in the thesis:
- **Clean prompts:** mean quality is flat across steps. The reference already has WER = 0 for 86% of utterances.
- **Noisy prompts are hard even for the reference:** Step 8 exceeds the 12.5% WER ceiling on 43.6% of them.
- **Noisy UTMOS sits at about 1.25 with almost no spread (SD 0.02).** The model copies the prompt's noise into the output, and UTMOS saturates at its floor. That makes UTMOS uninformative on noisy prompts, which is another reason it stays out of the failure rule.
- **Noisy mean WER is dominated by a few runaway transcripts.** The maximum per-utterance WER reaches 9,867% (looping or hallucinated output). Medians are 15.4% / 10.0% / 9.1% / 10.0% at Steps 4 / 5 / 6 / 8; the means look non-monotone only because of these outliers. This is why failures are counted per utterance rather than compared by means.
- **Failure causes** across the 3,000 non-reference evaluations: 232 WER-only, 453 SIM-only, 179 both.

---

## 4. Sway-grid figure (Figure 2.2)

![Sway grid and grid check](../figures/sway_grid_tswitch.png)

File: [`figures/sway_grid_tswitch.png`](../figures/sway_grid_tswitch.png). Generator: [`figures/generate_sway_grid_check.py`](../figures/generate_sway_grid_check.py), which uses only the formula and no data.

- **Panel (a)** follows the proposal's figure spec:
  - u on the x-axis, s = 0 (dashed) vs. s = −1 (solid);
  - the 17 teacher grid points;
  - t_switch = 0.25, with the teacher region shaded;
  - candidate Steps 4/5/6/8 with their NFE.
  - Annotation: 8 of 17 grid points lie at or below 0.25, and the first half of the grid (u ≤ 0.5) covers only t ≤ 0.293. This supports the TO DO point "(a) half of the grid points fall below t ≈ 0.29".
- **Panel (b)** shows the grid check from Section 1: the 16- vs. 32-interval grids near s = 0.25 and the stop index under each rule. Use it in Chapter 3 next to Remark 3.1, or drop it from Figure 2.2 if a single panel is preferred.
- The older [`figures/sway_grid_plot.png`](../figures/sway_grid_plot.png) is kept. It plots against the step index, has no t_switch line, and its Step 4/5 labels overlap, so the new figure is the one to use.

---

## 5. Proposal edits this implies

1. **Remark 3.1:** change "t₁₄ ≈ 0.236" to "t₁₄ ≈ 0.227". Cite `teacher_steps=32` in the released `generate_teacher_only` as support for the 32-interval reading.
2. **§2.3.1 cost paragraph:** 30 vs. 64 passes (2.13×); 32 vs. 64 with the endpoint evaluation; the paper's 2 student steps correspond to j = 2.
3. **§3.3.3:** 1,088 utterances, 666 prompt speakers; Table 3.4 sizes 100 / 200 / 200. Add the "prompt identity" caveat.
4. **Table 3.5 / §3.3.4:** name the actual SIM model (WavLM-Base+ SV), or re-run with WavLM-large.
5. **§2.6:** insert the WER / SIM / UTMOS / RTF text from Section 3.1.
6. **Results chapter:** cite RTF only from the full-pipeline timing run; label it T4; list the Ryzen-laptop runtime as not yet measured.
7. **Figure 2.2:** replace it with `sway_grid_tswitch.png`.
