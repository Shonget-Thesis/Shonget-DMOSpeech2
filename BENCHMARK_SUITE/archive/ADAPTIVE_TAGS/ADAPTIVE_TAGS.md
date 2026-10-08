# 🧠 Adaptive TAGS-Flow: Residual-Conditioned First-Correction Acceptance

**An Inference-Only Relaxation Rule for Teacher-Guided Accelerated Zero-Shot Speech Synthesis**  
**Document Code:** `SPEC-ADAPTIVE-TAGS-V2`  
**Core Innovation:** Residual-Conditioned First-Correction Acceptance via Bounded Monotonic Weighting  

---

## 📸 System Architecture

### 1. Sequential Relay Pipeline (Teacher ➔ Adaptive Switch ➔ Student)
![Adaptive TAGS-Flow Sequential Relay](ADAPTIVE_TAGS_FLOW_SEQUENTIAL_RELAY.png)

*(Also available: [Spacious 4-Stage Overview](ADAPTIVE_TAGS_FLOW_ARCHITECTURE_SPACIOUS.png) and [Compact Technical Layout](ADAPTIVE_TAGS_FLOW_ARCHITECTURE.png))*

---

## 🧭 1. Executive Summary & Rationale

### The Limitation of Static Acceptance ($w = 1.0$ or fixed $w = 0.85$):
In early prototypes of DMOSpeech 2's teacher-guided sampling, the student's first endpoint correction is fully accepted ($w = 1.0$) regardless of how far the student proposal moves from the teacher-derived endpoint. We previously tested a fixed partial acceptance ($w = 0.85$), which successfully reduced initial spectral flux but applied the same reduction whether the teacher and student strongly agreed or strongly disagreed.

However, real-world human speech is **highly diverse**:
* **Gentle Whispers:** Have low phonetic energy; small boundary displacement. The student's correction is likely highly compatible with the teacher's state.
* **Normal Conversational Speech:** Moderate displacement. Benefits from moderate relaxation to avoid high-frequency artifacts.
* **Loud, Fast, or Energetic Speech:** Strong vector displacement. A harsh handover collision here can cause phase tearing and spectral clicks.

**Adaptive TAGS-Flow** introduces an **inference-only, residual-conditioned acceptance rule**. It dynamically controls how much of the student's first endpoint correction is accepted based on its observed displacement from the teacher, without requiring retraining or additional neural-network evaluations.

---

## 📐 2. The Mathematical Formulation

### 2.1 The Relative Displacement Ratio ($r$)
At the handover step ($s_1 = 0.07612$), our inference engine observes two tensors:
1. $\mathbf{x}_{\text{teacher}}$: Extrapolated Teacher anchor state.
2. $\mathbf{x}_{\text{student}}$: Distilled Student generator proposal.

We measure the magnitude of the student's proposed correction relative to the teacher state. We define the **Relative Displacement Ratio ($r$)**:

$$r = \frac{\|\Delta \mathbf{x}\|_2}{\|\mathbf{x}_{\text{teacher}}\|_2} = \frac{\|\mathbf{x}_{\text{student}} - \mathbf{x}_{\text{teacher}}\|_2}{\|\mathbf{x}_{\text{teacher}}\|_2}$$

*Note: This acts as an inexpensive proxy for how aggressive the student's transition is, not necessarily a direct measure of "error".*

---

### 2.2 The Smooth Relaxation Weight ($w$)
We require an acceptance weight $w(r)$ that is:
- **Identity-preserving** when disagreement is zero ($w(0)=1$).
- **Monotonic and Smooth**, reducing acceptance as displacement grows.
- **Bounded**, to ensure we always accept a meaningful fraction of the student's correction.

We selected a shifted hyperbolic-tangent function because it satisfies these engineering requirements with a single interpretable gain parameter ($\kappa = 0.26$, selected via validation on pilot samples):

$$w_{\text{adaptive}}(r) = 1.0 - 0.26 \cdot \tanh(r)$$

The accepted endpoint $E$ passed to the remaining student steps becomes:

$$E = \mathbf{x}_{\text{teacher}} + w_{\text{adaptive}}(r) \cdot (\mathbf{x}_{\text{student}} - \mathbf{x}_{\text{teacher}})$$

Because $\tanh(r)$ is strictly bounded between $[0, 1)$ for $r \ge 0$, the acceptance weight is mathematically bounded: $w \in (0.74, 1.00]$. This bounds the *acceptance weight* (ensuring at least 74% of the student's update is applied), though it does not strictly bound the maximum physical displacement.

---

## 🧮 3. Casio fx-570CW Manual Calculation Walkthrough

You can verify the adaptive calculation live on your **Casio fx-570CW** in 3 simple steps (using a Sample 3 benchmark value where $\|\Delta \mathbf{x}\|_2 = 359.6839$ and $\|\mathbf{x}_{\text{teacher}}\|_2 = 544.9564$):

### 🔘 Step A: Compute the Displacement Ratio ($r$)
$$r = \frac{359.6839}{544.9564}$$
* **Calculator Display:** `0.6600`

### 🔘 Step B: Compute $\tanh(r)$
#### Using the `[CATALOG]` Key on fx-570CW:
1. Press **`[CATALOG]`** ➔ **`Hyperbolic Func`** ➔ **`tanh`**
2. Press **`[Ans]`** ➔ **`[EXE]`**
* **Calculator Display:** `0.5784`

### 🔘 Step C: Compute Dynamic Acceptance Weight ($w_{\text{adaptive}}$)
$$w = 1.0 - 0.26 \times 0.5784$$
* **Calculator Display:** `0.8496` (Accepts ~85% of the student's correction)

---

## 💻 4. Python Implementation (`smooth_dmospeech.py`)

Here is the exact production code for `proposed_system/smooth_dmospeech.py`:

```python
# ==============================================================================
# 🌟 PROPOSED CONTRIBUTION: Residual-Conditioned First-Correction Acceptance
# ==============================================================================
if idx == 0:
    delta = output - x1_teacher
    
    # 1. Measure live tensor norms across [1, T, D]:
    norm_delta = torch.norm(delta)
    norm_teacher = torch.norm(x1_teacher) + 1e-8
    
    # 2. Compute Relative Displacement Ratio r:
    ratio = (norm_delta / norm_teacher).item()
    
    # 3. Dynamic Bounded Hyperbolic Tangent Formulation:
    # w adapts smoothly between [0.74, 1.00]
    w_adaptive = float(np.clip(1.0 - 0.26 * np.tanh(ratio), 0.74, 1.00))
    
    # 4. Surgical Handover Relaxation:
    x1 = x1_teacher + w_adaptive * delta
    
    if verbose:
        print(f"   [Adaptive Handover Applied]")
        print(f"      Displacement Ratio r = {ratio:.4f}  -->  Acceptance w = {w_adaptive:.4f}")
else:
    x1 = output
```

---

## 🎓 5. Defense Q&A: Master Answers for Thesis Panelists

### Question 1: *"Why did you believe this would work before implementing it?"*
> **Answer:**  
> *"DMOSpeech 2 showed that teacher-guided early sampling and student-based refinement can provide a useful quality–efficiency trade-off. However, its first student correction is fully accepted regardless of how far the student proposal moves from the teacher-derived endpoint. We hypothesized that this correction magnitude could serve as a low-cost indicator of handover aggressiveness. We therefore introduced an inference-only relaxation rule that preserves the baseline when teacher and student agree, and progressively limits the accepted fraction as their displacement increases. This provided a testable hypothesis that residual-dependent partial acceptance could improve the quality–efficiency trade-off compared with full acceptance."*

### Question 2: *"What is the exact novelty of your work?"*
> **Answer:**  
> *"The novelty is not $\tanh$, interpolation, or teacher–student collaboration individually. Our contribution is the application and evaluation of a residual-conditioned, inference-only first-correction acceptance mechanism for frozen teacher-guided accelerated TTS. Unlike hard timestep switching, learned expert routing, or methods that invoke the teacher adaptively (e.g., CVPR 2024), our mechanism directly controls how much of the student's first endpoint correction is accepted, without retraining or extra neural evaluations."*

### Question 3: *"Why did you choose the $\tanh$ function?"*
> **Answer:**  
> *"We selected a bounded hyperbolic tangent because it is smooth, monotonic, equals one at zero displacement, and introduces no additional neural-model evaluations. It naturally saturates, mathematically guaranteeing that the acceptance weight never drops below $1.0 - 0.26 = 0.74$. This ensures we always apply at least 74% of the student's update, preventing the output from stalling or becoming overly damped."*
