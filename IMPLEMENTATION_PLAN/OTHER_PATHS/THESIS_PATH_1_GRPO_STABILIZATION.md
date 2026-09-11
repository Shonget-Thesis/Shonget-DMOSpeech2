# 🎓 Thesis Research Proposal: Stabilizing GRPO for Duration Prediction in Flow-Matching Speech Synthesis

> **Target Split:** 70% Mathematical Theory / 30% Empirical Implementation  
> **Base Codebase:** [DMOSpeech 2](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202) & [F5-TTS](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/f5_tts)  
> **Status:** Open Research Gap (Unsolved in current peer-reviewed literature)

---

## 📌 Executive Summary (The 30-Second Elevator Pitch)

* **The System:** **DMOSpeech 2** is a state-of-the-art AI system that generates ultra-fast human-like speech from text.
* **The Breakthrough:** It is the first system to use **GRPO (Group Relative Policy Optimization)**—the same reinforcement learning technique used in DeepSeek—to teach an AI how long each sentence should take to speak (duration prediction).
* **The Unsolved Problem:** In the published paper, training GRPO past **1,500 steps** causes the model to overfit, become unstable, and collapse (destroying speech quality). The authors had to stop training early because they had no fix.
* **Your Thesis:** You will formulate the **mathematical theory of why GRPO collapses in discrete speech duration modeling**, derive variance bounds, and invent a **Stabilized / Adaptive Trust-Region GRPO algorithm** that trains stably to 2,500–5,000+ steps.

---

## 📖 Part 1: The "Zero to Hero" Primer (TTS & RL in Plain English)

If you have never worked with speech synthesis or reinforcement learning before, here are the core concepts explained in simple terms:

```
[Target Text] ───► [Duration Predictor (RL/GRPO)] ───► Determines Total Time (e.g., 3.4 seconds)
                             │
                             ▼
[Prompt Audio] ──► [Flow Matching / Diffusion (DiT)] ──► Mel-Spectrogram ──► [Vocos] ──► Final Audio (.wav)
```

---

### 1. How does modern AI make sound? (Mel-Spectrograms & Vocoders)
* Computers do not generate raw sound waves directly because raw audio contains 24,000 numbers per second (too heavy to predict).
* Instead, AI generates a **Mel-Spectrogram** (a visual heat-map of sound frequencies over time—like digital sheet music).
* A separate neural network called a **Vocoder** (like [Vocos](https://github.com/gemelo-ai/vocos)) reads this spectrogram image and converts it into real `.wav` audio.

---

### 2. What is "Flow Matching" (Diffusion)?
* Imagine starting with a picture of pure random TV static (Gaussian noise).
* **Flow Matching** is a continuous mathematical recipe that calculates the velocity vector field to transform that random noise step-by-step into clear, beautiful speech spectrograms.
* **F5-TTS** and **DMOSpeech 2** use Flow Matching to generate speech that matches the voice and emotion of your reference audio clip.

---

### 3. Why is "Duration" a huge problem in Diffusion TTS?
* Autoregressive speech models (like GPT-based voice models) produce speech word-by-word and stop whenever they feel like it.
* But Diffusion / Flow-Matching models **must know the exact total length of the audio canvas before they start drawing sound**.
* If the predicted duration is **too short**, the audio sounds rushed, words get cut off, and speech becomes garbled.
* If the predicted duration is **too long**, the audio sounds unnaturally dragged out, with awkward silences or hallucinated repetitions.
* Therefore, the system needs a separate **Duration Predictor** ([`SpeechLengthPredictor`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/duration_predictor.py#L34)).

---

### 4. What is Reinforcement Learning (RL) in simple terms?
* Instead of telling the AI the exact mathematical answer for every situation (Supervised Learning), RL lets the AI try several guesses, grades the results using a **Reward Function** (score), and tells the AI: *"Do more of what got high scores, and avoid what got low scores."*

---

### 5. What is GRPO (Group Relative Policy Optimization)?
* In older RL methods (like PPO), you had to train two giant AI models:
  1. An **Actor** (the model making decisions).
  2. A **Critic** (a separate referee model estimating the value of each action).
* **GRPO is a lightweight upgrade:** It completely removes the Critic referee.
* **How GRPO works:**
  1. For a single sentence, the Duration Predictor generates **$K$ different duration guesses** (e.g., $K=16$ guesses: 3.1s, 3.4s, 2.9s, 3.8s...).
  2. The speech engine generates audio for all $K$ guesses.
  3. A reward function grades all $K$ audio clips based on **Intelligibility (CTC Error Rate)** and **Speaker Voice Match (Cosine Similarity)**.
  4. The model scores are **normalized relative to the group average**:
     $$\text{Advantage}_i = \frac{\text{Reward}_i - \text{Mean(Group Rewards)}}{\text{Std(Group Rewards)}}$$
  5. The model adjusts its weights to favor durations that scored above the group average.

---

## 💥 Part 2: The Exact Problem (The 1.5K Step Cliff)

In the DMOSpeech 2 paper ([arXiv:2507.14988](https://arxiv.org/abs/2507.14988), Appendix A.2.1), the authors discovered a major roadblock:

```
Speech Quality
     ▲
     │          Peak Performance (~1,500 Steps)
     │                ╭────────╮
     │               ╱          ╲
     │              ╱            ╲   CRASH / COLLAPSE
     │             ╱              ╲  (Speech becomes robotic,
     │  ──────────╯                ╲  words get cut off)
     │  (Supervised)                ▼
     └────────────────────────────────────────► Training Steps
     0            500    1,000   1,500   2,000
```

### Why does it collapse after 1,500 steps? (The 3 Root Causes)

1. **Noisy Group Baseline (No Critic):**
   * When group size $K$ is small ($K=8$ or $16$), the group average is noisy. If an outlier sample gets a high reward by pure luck, the model takes an erratic, giant gradient step in the wrong direction.
2. **Division by Zero / Reward Homogeneity:**
   * If all $K$ duration guesses sound roughly the same, $\text{Std}(\text{Rewards}) \approx 0$. Dividing by zero or near-zero creates exploding updates.
3. **Reward Hacking (Goodhart's Law):**
   * The reward only measures proxy numbers: Connectionist Temporal Classification (CTC) loss and Speaker Verification (SV) cosine distance.
   * After 1,500 steps, the AI discovers "cheats" (e.g., clipping off trailing syllables to artificially speed up CTC or flattening pitch to game the speaker embedding). It maximizes the reward score while producing speech that sounds unlistenable to humans.

---

## 🎯 Part 3: The Proposed Thesis

### Working Title:
> **"Variance Bounds and Trust-Region Stabilization of Group Relative Policy Optimization for Non-Differentiable Duration Modeling in Flow-Matching Speech Synthesis"**

---

### 🏛️ The 70% Theory Breakdown (Mathematical Contributions)

1. **Advantage Variance Proofs under Finite Group Sampling:**
   * Formalize duration prediction as a contextual bandit / Markov decision policy $\pi_\phi(L \mid x, p)$.
   * Prove mathematically that the empirical advantage variance $\text{Var}(\hat{A}_i)$ scales as $\mathcal{O}(1/K)$ and diverges when reward diversity $\Delta r \to 0$.
2. **Policy Drift Bounds under Surrogate Reward Landscapes:**
   * Prove why a static KL divergence penalty ($\beta D_{\text{KL}}$) is mathematically insufficient to bound policy drift on non-convex proxy reward surfaces over long training horizons ($T > 1500$).
3. **Derivation of Stabilized / Adaptive Trust-Region GRPO (TR-GRPO):**
   * Derive an analytically grounded **Adaptive KL Schedule ($\beta_t$)** based on empirical Fisher Information / KL divergence rates:
     $$\beta_{t+1} = \beta_t \cdot \exp\left( \alpha \cdot (D_{\text{KL}}(\pi_t \parallel \pi_{\text{ref}}) - \delta_{\text{target}}) \right)$$
   * Derive a **Variance-Damped Advantage Estimator ($\hat{A}_i^*$)** that includes an analytical prior regularizer to prevent division by near-zero variance.
   * Prove a **Monotonic Improvement Theorem** showing that the expected true speech quality is non-decreasing under this stabilized objective.

---

### 💻 The 30% Empirical Breakdown (Implementation & Experiments)

1. **Implement TR-GRPO in Code:**
   * Modify [`src/grpo_duration_trainer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py) with the dynamic KL controller, duration-prior regularized reward, and annealed clipping parameter ($\epsilon: 0.20 \to 0.10$).
2. **Ablation Studies & Scaling Verification:**
   * Train models at 500, 1000, 1500, 2000, 3000, and 5000 steps.
   * Compare:
     * *Vanilla GRPO* (Baseline — collapses past 1.5k)
     * *Your Stabilized GRPO* (Maintains high performance past 3k–5k steps)
3. **Evaluation Metrics:**
   * **Objective:** Word Error Rate (Whisper / Paraformer), Speaker Cosine Similarity (WavLM / WeSpeaker), and UTMOS.
   * **Subjective:** Human Comparative Mean Opinion Score (CMOS) tests verifying that naturalness does not degrade.

---

## 🗺️ Part 4: Codebase Navigation Guide

Here are the key files in this repository where the algorithms live:

| File | What it does | Relevance to your thesis |
| :--- | :--- | :--- |
| [`src/grpo_duration_trainer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py) | GRPO training loop, advantage calculation, reward gathering | **Core file you will modify and benchmark.** |
| [`src/duration_predictor.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/duration_predictor.py) | Transformer cross-attention architecture for speech length | The neural network policy $\pi_\phi$. |
| [`src/infer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/infer.py) | Inference pipeline, duration clamping, teacher-student sampling | Testbed where you generate audio with your duration model. |
| [`src/guidance_model.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/guidance_model.py) | Reward models (CTC Loss + ECAPA-TDNN Speaker Verification) | Contains the proxy reward definitions. |
| [`src/f5_tts/model/cfm.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/f5_tts/model/cfm.py) | Conditional Flow Matching core logic and ODE solver | The continuous flow engine that converts duration into audio. |

---

## 📚 Part 5: Step-by-Step Study Roadmap for the Student

To get up to speed quickly, read these materials in order:

### Week 1: Speech & Diffusion Basics
1. **Mel Spectrograms:** Understand how audio is converted to frequency bins.
2. **F5-TTS Paper ([arXiv:2410.06885](https://arxiv.org/abs/2410.06885)):** Understand how text + reference audio infills missing speech via Flow Matching.

### Week 2: Reinforcement Learning & GRPO
1. **PPO Paper (Schulman et al., 2017):** Understand clipping and policy ratios.
2. **DeepSeekMath / GRPO Paper ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)):** Understand how group relative advantage works without a critic.

### Week 3: The Target Paper
1. **DMOSpeech 2 Paper ([arXiv:2507.14988](https://arxiv.org/abs/2507.14988)):** Read Sections 3.2 (GRPO for Duration) and Appendix A.2 (Hyperparameters & Instability).

---

## 📝 Next Steps
* Keep this document as your master reference sheet.
* Whenever you are ready to start the theoretical derivations or modify [`src/grpo_duration_trainer.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/src/grpo_duration_trainer.py), we can write the mathematical equations and code step by step!
