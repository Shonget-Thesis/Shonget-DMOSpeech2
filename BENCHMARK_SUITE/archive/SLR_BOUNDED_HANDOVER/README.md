# 📐 SLR Bounded Handover: Teacher-Student Inferencing Framework

**Document Code:** `SLR-BOUNDED-HANDOVER-V1`  
**Direct Academic Source:** `SLR_for_Teacher_Student_Inferencing.pdf` (Section 9: *Recommended formulation and derivation*, pp. 21–24)  
**Input Dataset:** Uses the exact 5 reference prompts and transcripts from [`ADAPTIVE_TAGS/input/`](../ADAPTIVE_TAGS/input/INPUT_MANIFEST.md).

---

## 🧭 1. Executive Summary & Formulation

In accelerated zero-shot TTS inference ([DMOSpeech 2](https://arxiv.org/abs/2507.14988)), the pipeline combines a high-capacity un-distilled **Teacher** ($t = 0 \to 0.07$) with an efficient distilled **Student** ($t = 0.07 \to 1.00$).

### The Inspected Baseline Mechanism
As established in Section 4 of the SLR:
1. The teacher integrates via Euler to its stopping time $\tau \approx 0.076$ and extrapolates a clean **endpoint estimate**:
   $$T = x_\tau + (1 - \tau)\mathbf{v}_T(x_\tau, \tau, c)$$
2. The student draws fresh stochastic noise $\epsilon_1 \sim \mathcal{N}(0, I)$ and constructs its noisy intermediate state:
   $$z_1 = (1 - t_1)\epsilon_1 + t_1 T$$
3. The student evaluates its generator to propose a clean reconstruction:
   $$S = z_1 + (1 - t_1)\mathbf{v}_S(z_1, t_1, c)$$
4. The proposed displacement vector is:
   $$D = S - T$$

---

## 🔬 2. The SLR Bounded Displacement Cap (Section 9)

Rather than relying on ungrounded heuristic functions or overclaiming Lyapunov BIBO stability, Section 9 derives the handover as **Constrained Closest-Proposal Selection (Euclidean Ball Projection)**:

Let the allowed endpoint set be a closed ball of radius $\rho$ centered at the teacher anchor:
$$\mathcal{B}(T, \rho) = \left\{ Y : \|Y - T\|_2 \le \rho \sqrt{d} \right\}$$

The optimal accepted endpoint $E$ minimizes distance to the student proposal $S$:
$$E = \arg\min_{Y \in \mathcal{B}(T, \rho)} \frac{1}{2} \|Y - S\|_2^2$$

### The Resulting Projection Rule:
Compute the root-mean-square displacement per coordinate:
$$r = \text{RMS}(D) = \frac{\|D\|_2}{\sqrt{d}}$$

The acceptance weight $\lambda_\rho(D)$ is:
$$\lambda_\rho(D) = \begin{cases} 1.0, & r \le \rho \\ \frac{\rho}{r}, & r > \rho \end{cases} = \min\left(1, \frac{\rho}{r}\right)$$

$$E = T + \lambda_\rho(D)(S - T)$$

### Mathematical Guarantees Proved in the SLR:
* **Proposition 1 (Displacement Bound):** $\text{RMS}(E - T) = \min(r, \rho) \le \rho$.
* **Proposition 2 (Withheld Correction):** $\text{RMS}(E - S) = \max(0, r - \rho)$.
* **Proposition 3 (Nonexpansiveness):** $\|P_\mathcal{B}(S_1) - P_\mathcal{B}(S_2)\|_2 \le \|S_1 - S_2\|_2$.
* **Zero Latency Penalty:** Vector arithmetic on already-computed tensors adds $0$ extra neural forward calls ($< 1$ ms).

---

## ⚖️ 3. Comparative 4-Mode Evaluation Matrix

This module implements the 4 comparison modes recommended in SLR Section 11:

| Mode ID | Formulation | Status in Thesis |
| :--- | :--- | :--- |
| `baseline` | $E = S$ ($\lambda = 1.00$) | Unconstrained default DMOSpeech 2 |
| `fixed` | $E = T + 0.85(S - T)$ | Static relaxation baseline (P2 report) |
| `tanh` | $w = 1.0 - 0.26\tanh\left(\frac{\|D\|_2}{\|T\|_2}\right)$ | Your initial smooth dynamic formulation |
| `slr_cap` | $\lambda = \min\left(1, \frac{\rho}{\text{RMS}(D)}\right)$ | **SLR Section 9 Recommended Euclidean Projection** |
| `cosine` | $w = \frac{1 - \cos(\pi t_1)}{2} \approx 0.0142$ | Ablation proving why time schedules fail at sparse stages |

---

## 🚀 4. How to Run

### Single Utterance Test (Mirroring `ADAPTIVE_TAGS/run_custom_test.py`):
```bash
# Run with SLR Bounded Cap (default rho=1.5)
python SLR_BOUNDED_HANDOVER/run_slr_custom_test.py --mode slr_cap --rho 1.5

# Run with your original tanh formula
python SLR_BOUNDED_HANDOVER/run_slr_custom_test.py --mode tanh

# Run with custom audio and prompt
python SLR_BOUNDED_HANDOVER/run_slr_custom_test.py \
  --audio ADAPTIVE_TAGS/input/input_1_female_nature.wav \
  --ref_text "Some call me nature. Others call me Mother Nature." \
  --text "Synthesizing with mathematically bounded Euclidean projection." \
  --mode slr_cap
```

### Full 5-Voice Comparative Benchmark:
```bash
python SLR_BOUNDED_HANDOVER/batch_compare_5_samples.py
```
* Generates paired comparisons across all 5 voices in `ADAPTIVE_TAGS/input/`.
* Evaluates Baseline, Fixed, Tanh, and SLR Cap with **strict noise-seed sharing** (`noise_seed=42`).
* Outputs formatted markdown table and `benchmark_results.json`.
