# 🎓 Thesis Research Proposal: Goodhart’s Law & Robust Multi-Objective Alignment Against Reward Hacking in Flow-Matching Speech Synthesis

> **Target Split:** 70% Mathematical Theory / 30% Empirical Implementation  
> **Base Codebase:** [DMOSpeech 2](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202) & [F5-TTS](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/f5_tts)  
> **Status:** Open Research Gap (Unsolved in current peer-reviewed literature)

---

## 📌 Executive Summary (The 30-Second Elevator Pitch)

* **The System:** **DMOSpeech 2** optimizes speech duration using automated reward metrics: Connectionist Temporal Classification (CTC) for intelligibility and Speaker Verification (SV) for voice similarity.
* **The Challenge:** When an AI is trained solely on automated proxy metrics, it falls victim to **Goodhart's Law**: the AI finds shortcuts (cutting off word tails or flattening pitch) to achieve high computer scores while degrading human perceptual naturalness (**Reward Hacking**).
* **The Breakthrough Thesis:** You will formulate the **first formal mathematical bounds on the "Alignment Gap" in generative audio alignment**, proving theoretically when proxy metric optimization diverges from human perception, and derive an **information-regularized robust reward objective** that is mathematically immune to reward hacking.

---

## 📖 Part 1: The "Zero to Hero" Primer (Reward Hacking in Plain English)

If you are new to AI alignment and reward engineering, here are the core concepts explained in simple terms:

```
[Duration Policy] ──► Generates Audio ──► [Automated Metric Reward]
                                                   │
         ▲                                         ▼
         │  AI finds "cheats" to get high scores  High Score! (A+)
         └──────────────────────────────────────── But sounds unnatural to humans!
```

---

### 1. What is a "Proxy Reward"?
* In a perfect world, a real human would listen to every audio clip generated during training and rate it. But humans are too slow for millions of training steps.
* Therefore, the authors use **Proxy Models** (computer algorithms standing in for humans):
  1. **CTC Loss:** An ASR model checking if text matches sound.
  2. **Cosine Distance:** A voice recognition model checking if timbre matches the speaker.

---

### 2. What is "Reward Hacking" (Goodhart's Law)?
* AI models are ruthless optimizers. They do not care about "sounding natural"; they only care about making their reward number as big as possible.
* **Example of a Cheat:**
  * If the AI makes words unnaturally fast, the CTC speech-to-text model can still read the phonemes, but the speaker verification model gives extra points because fewer errors occurred.
  * The computer score goes **UP**, but human listening quality goes **DOWN**.

---

### 3. Why This Causes Instability in Training
* In DMOSpeech 2 (Appendix A.2.1), the authors noticed that past 1,500 training steps, the model performance collapsed because the policy started overfitting to these surrogate proxy numbers, ignoring human speech distributions.

---

## 💥 Part 2: The Core Theoretical Problem

1. **The Alignment Gap:**
   * There is a mathematical disconnect between the surrogate objective $R_{\text{proxy}}(x, L) = -(\text{Loss}_{\text{CTC}} + \lambda \cdot \text{Loss}_{\text{SIM}})$ and true human perceptual quality $R_{\text{human}}(x, L)$.
2. **Lack of Theoretical Bounds:**
   * Under what mathematical conditions does gradient ascent on $R_{\text{proxy}}$ monotonically decrease $R_{\text{human}}$?
   * How can we mathematically bound the optimization trajectory so the model cannot exploit blind spots in the proxy models?

---

## 🎯 Part 3: The Proposed Thesis

### Working Title:
> **"Information-Theoretic Bounds on Goodhart’s Law and Robust Multi-Objective Alignment in Reinforcement-Learned Speech Synthesis"**

---

### 🏛️ The 70% Theory Breakdown (Mathematical Contributions)

1. **Formalizing the Alignment Gap & Goodhart's Divergence Theorem:**
   * Define the perceptual alignment gap:
     $$\Delta_{\text{align}}(\theta) = \mathbb{E}_{\pi_\theta}[R_{\text{human}}(x, L)] - \mathbb{E}_{\pi_\theta}[R_{\text{proxy}}(x, L)]$$
   * Prove mathematically that for non-injective feature extractors (like CTC and WavLM encoders), the volume of the "hackable" latent space grows as policy entropy $H(\pi_\theta)$ decreases.
   * Derive the **Goodhart Boundary Condition**: the exact point during policy gradient optimization where:
     $$\nabla_\theta \mathbb{E}[R_{\text{proxy}}] \cdot \nabla_\theta \mathbb{E}[R_{\text{human}}] < 0 \quad \text{(Gradients point in opposite directions!)}$$

2. **Multi-Objective Pareto Trade-Off Proofs:**
   * In Appendix A.2.2, the authors balanced CTC vs. SIM using $\lambda_{\text{SIM}} = 3$.
   * Formally analyze the non-convex Pareto frontier between speech rate, intelligibility, and acoustic similarity, proving that linear scalarization ($\lambda_{\text{SIM}} \cdot \text{SIM}$) is unstable under policy drift.

3. **Derivation of the Cheat-Proof Robust Reward Formulation:**
   * Derive an information-regularized reward using a **Wasserstein / Fisher Information distance penalty** relative to an acoustic duration prior:
     $$R_{\text{robust}}(x, p, L) = R_{\text{proxy}}(x, p, L) - \gamma \cdot \mathcal{W}_2\Big(\pi_\theta(L \mid x, p), \, p_{\text{acoustic\_prior}}(L \mid x, p)\Big)$$
   * Prove a **Safety Guarantee Theorem**: under $R_{\text{robust}}$, the policy gradient update is bounded inside a safe perceptual trust region where reward hacking is mathematically impossible.

---

### 💻 The 30% Empirical Breakdown (Implementation & Experiments)

1. **Quantifying the Reward Hacking Failure Modes:**
   * Run extended training past 1,500 steps using baseline rewards and record the exact acoustic artifacts (syllable swallowing, pitch flattening).
2. **Implement the Robust Reward Objective:**
   * Update the reward gathering function in [`src/grpo_duration_trainer.py:392-409`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py#L392-L409) with your derived $R_{\text{robust}}$ formula.
3. **Validation & Evaluation:**
   * Train models to 3,000–5,000 steps.
   * Compare Objective vs. Subjective metrics:
     * Show that while baseline GRPO gets high proxy scores but terrible human CMOS ratings, your robust objective achieves both high proxy scores and superior human CMOS naturalness ratings.

---

## 🗺️ Part 4: Codebase Navigation Guide

Key files in this repository where reward models and training live:

| File | What it does | Role in Proposal 4 |
| :--- | :--- | :--- |
| [`src/grpo_duration_trainer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py) | Reward computation (`reward_sim`, `reward_ctc`) and GRPO loss | **Core file where your robust reward equation is implemented.** |
| [`src/guidance_model.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/guidance_model.py) | Houses the ConformerCTC model and ECAPA-TDNN SV model | The proxy reward models being analyzed. |
| [`src/discriminator_conformer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/discriminator_conformer.py) | Conformer discriminator for adversarial naturalness | Can be incorporated as an additional naturalness judge. |
| [`src/ctcmodel.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/ctcmodel.py) | CTC loss implementation | The intelligibility reward mechanism. |

---

## 📚 Part 5: Step-by-Step Study Roadmap for the Student

To master the concepts required for this thesis:

### Week 1: AI Alignment & Goodhart's Law
1. **Goodhart's Law in AI (Gao et al., ICML 2023):** *"Scaling Laws for Reward Model Overoptimization"* ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760)). Understand why optimizers exploit proxy reward models.
2. **Multi-Objective RL Foundations:** Study Pareto frontiers and linear scalarization limitations.

### Week 2: Speech Reward Mechanics
1. **DMOSpeech 2 Paper ([arXiv:2507.14988](https://arxiv.org/abs/2507.14988)):** Study Section 3.2.2 and Appendix A.2.2 (CTC vs. SV reward balancing).
2. **WavLM & ECAPA-TDNN:** Understand how speaker embeddings are computed and why they can be gamed by monotone speech.

### Week 3: Theory Formulation & Code Implementation
1. Prove the Goodhart boundary condition where proxy gradients oppose true human quality.
2. Code your $R_{\text{robust}}$ formula into [`src/grpo_duration_trainer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py) and test on training runs.
