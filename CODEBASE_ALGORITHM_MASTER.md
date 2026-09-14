# 📘 MASTER PROJECT SPECIFICATION & COMPLETE CODEBASE ARCHITECTURE
## Project: DMOSpeech 2 & TAGS-Flow Inference Acceleration Framework
**Date:** September 2026  
**Document Purpose:** Self-contained, comprehensive technical blueprint and code reference. An incoming AI model or researcher can read this single file to fully understand every algorithm, neural architecture, training paradigm, inference procedure, and mathematical derivation in this repository without needing to inspect external files.

---

# Table of Contents
1. [High-Level Project Executive Summary](#1-high-level-project-executive-summary)
2. [Complete Repository Directory Map](#2-complete-repository-directory-map)
3. [End-to-End Pipeline & Inference Execution Flow](#3-end-to-end-pipeline--inference-execution-flow)
4. [Acoustic Core Architectures (DiT, Guidance, UniModel, Modules)](#4-acoustic-core-architectures)
5. [Duration Prediction & Reinforcement Learning (SLP, GRPO, Reward Hacking)](#5-duration-prediction--reinforcement-learning)
6. [Distribution Matching Distillation (DMD Trainer, Discriminator, CTC)](#6-distribution-matching-distillation)
7. [Inference Engine Breakdown (`infer.py`) & The Two Critical Bugs](#7-inference-engine-breakdown--the-two-critical-bugs)
8. [The TAGS-Flow Mathematical Solution & Theoretical Proofs](#8-the-tags-flow-mathematical-solution--theoretical-proofs)
9. [TAGS-Flow Implementation Details (Drop-in Code Replacement)](#9-tags-flow-implementation-details)
10. [Ablation Study, Benchmarking & Experimental Verification](#10-ablation-study-benchmarking--experimental-verification)
11. [Master Hyperparameter Reference Table](#11-master-hyperparameter-reference-table)

---

# 1. High-Level Project Executive Summary

### What is DMOSpeech 2?
**DMOSpeech 2** is a state-of-the-art, metric-optimized zero-shot Text-to-Speech (TTS) framework developed by Yinghao Aaron Li et al. (July 2025), built upon **F5-TTS** (Oct 2024).

Standard diffusion and flow-matching TTS systems (such as F5-TTS, Voicebox, E2-TTS) require 32 to 64 Ordinary Differential Equation (ODE) steps to numerically transport noise $x_0 \sim \mathcal{N}(0, I)$ to clean speech mel-spectrograms $x_1$. DMOSpeech 2 addresses two major bottlenecks in modern zero-shot TTS:
1. **Inference Acceleration via Distribution Matching Distillation (DMD):** Distills a 32-step teacher DiT (Diffusion Transformer) into a 1-to-4 step student generator (`UniModel`) using adversarial guidance and distribution matching losses.
2. **Metric Optimization via Reinforcement Learning:** Unlike prior works that used heuristic rate-based formulas or simple supervised duration predictors, DMOSpeech 2 trains a **Speech Length Predictor (SLP)** using **Group Relative Preference Optimization (GRPO)** to optimize downstream speech quality metrics (Word Error Rate via ASR and Speaker Similarity via ECAPA-TDNN).
3. **Teacher-Guided Sampling:** To combat the mode collapse and loss of acoustic detail inherent in 4-step distilled students, DMOSpeech 2 runs the teacher model for the initial fraction of sampling (up to $t = 0.07$ or $0.30$) before handing off to the fast student.

### What is TAGS-Flow?
**TAGS-Flow (Trajectory-Aware Guided Sampling)** is the thesis framework introduced to solve the two mathematical failure modes discovered in DMOSpeech 2:
1. **$\mathcal{C}^0$ Boundary Collision:** In DMOSpeech 2, the teacher ODE terminates abruptly at $t_{\text{stop}}$, and a single-step linear extrapolation is handed over to the student. Because the student's velocity field is trained under a different marginal distribution, their velocity vectors collide ($\angle(v_T, v_S) \neq 0$), injecting high-frequency phase roughness and acoustic distortion. TAGS-Flow introduces a **cosine transition manifold $w(t)$** guaranteeing continuous first derivatives ($\mathcal{C}^1$ continuity) and zero boundary collision.
2. **Compounding Duration Misallocation & Cadence Drift:** Flow-matching generates mel frames over a rigid, pre-allocated sequence length $T$. When the neural SLP misestimates $T$, speech either stretches into slow robotic drones or compresses into rushed slurs. TAGS-Flow formulates a training-free **Cadence-Scaling factor ($\alpha$)** that scales the velocity vector based on the ratio between the reference speaker's natural cadence and the predicted cadence.

---

# 2. Complete Repository Directory Map

```
Shonget-DMOSpeech2/
├── README.md                            # Upstream DMOSpeech 2 documentation
├── requirements.txt                     # Dependencies (torch, torchaudio, accelerate, vocos, etc.)
├── demo.ipynb                           # Interactive Jupyter demo for testing inference
├── thesis_path_5 (1).md                 # TAGS-Flow thesis blueprint and specification
├── TAGS_FLOW_CONTEXT_BRIEF.md           # Onboarding brief summarizing the changes
├── CONVERSATION_HISTORY.md              # Dialogue record of technical decisions
│
├── data/                                # Training corpora, metadata, and tokenizers
│
└── src/                                 # Primary Source Code
    ├── infer.py                         # [CRITICAL] Full inference engine: DMOInference class
    ├── duration_predictor.py            # [CRITICAL] SpeechLengthPredictor (SLP) neural architecture
    ├── unimodel.py                      # [CRITICAL] Wrapper combining Teacher DiT and Student DiT
    ├── guidance_model.py                # Guidance module providing real/fake score updates (DMD)
    ├── grpo_duration_trainer.py         # RL trainer for SLP using Group Relative Preference Optimization
    ├── duration_trainer.py              # Supervised duration trainer (L1 and Cross-Entropy)
    ├── duration_trainer_with_prompt.py  # Supervised duration trainer with reference acoustic prompts
    ├── dmd_trainer.py                   # Distillation trainer using DMD and adversarial objectives
    ├── discriminator_conformer.py       # Conformer-based discriminator for adversarial speech loss
    ├── ctcmodel.py                      # Conformer-CTC model for phonetic alignment & ASR loss
    ├── ecapa_tdnn.py                    # ECAPA-TDNN speaker embedding extractor for similarity reward
    │
    └── f5_tts/                          # Backbone library (F5-TTS primitives)
        ├── api.py                       # High-level API helpers
        └── model/
            ├── cfm.py                   # Conditional Flow Matching (OT-CFM) sampling & odeint
            ├── modules.py               # MelSpec (Vocos/BigVGAN), TimestepEmbedding, DiT blocks
            ├── dataset.py               # DynamicBatchSampler, collate_fn, audio preprocessing
            ├── utils.py                 # Tokenizer mapping, padding, masks, seed helpers
            └── backbones/
                ├── dit.py               # DiT (Diffusion Transformer) with Rotary Embeddings
                └── unett.py             # UNetT alternative backbone
```

---

# 3. End-to-End Pipeline & Inference Execution Flow

During inference, audio is synthesized through four sequential stages:

```
[Input Text + Reference Audio Prompt]
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 1: Duration Prediction (SpeechLengthPredictor)   │
 │ • Extracts reference Mel from prompt audio (24kHz)     │
 │ • Tokenizes target & reference text to Pinyin IDs      │
 │ • Encodes Text + Cross-Attends Mel                     │
 │ • Outputs discrete duration class: [0, 300] × 10 frames│
 └────────────────────────────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 2: Initial Noise & Conditioning Preparation      │
 │ • Concatenates prompt mel + silence budget (pad to T)  │
 │ • Generates initial noise y_0 ~ N(0, I)                │
 │ • Builds binary masking cond_mask                      │
 └────────────────────────────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 3: Neural Flow Integration (ODE Solver)          │
 │ • Vanilla DMOSpeech 2: Teacher ODE -> Student Steps    │
 │ • TAGS-Flow: Blended ODE via w(t) with Cadence Scale α │
 │   x_{t+h} = x_t + h * (α * v_hybrid(x_t, t))           │
 └────────────────────────────────────────────────────────┘
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 4: Vocoder Inversion (Vocos)                     │
 │ • Slices out generated mel segment (removes prompt)    │
 │ • Rescales by model checkpoint scale factor            │
 │ • Vocos neural vocoder: Mel (100-dim) -> 24kHz Waveform│
 └────────────────────────────────────────────────────────┘
```

### Top-Level Invocation (`demo.ipynb` & `infer.py`)
```python
from infer import DMOInference

# Initialize the engine
tts = DMOInference(
    student_checkpoint_path="model_85000.pt",
    duration_predictor_path="model_1500.pt",
    model_type="F5TTS_Base",
    num_student_step=4,
    device="cuda"
)

# Run zero-shot generation
waveform = tts.generate(
    audio_path="reference.wav",
    prompt_text="Today is a beautiful sunny morning.",
    gen_text="Artificial intelligence has transformed zero-shot speech synthesis.",
    teacher_stopping_time=0.07,
    student_start_step=0,
    speed=1.0,
    cfg_strength=1.0
)
```

---

# 4. Acoustic Core Architectures

### 4.1 Diffusion Transformer (`src/f5_tts/model/backbones/dit.py`)
The acoustic backbone is an autoregressive-free **Diffusion Transformer (DiT)** operating on 100-channel mel-spectrograms.
* **Architecture Details (`F5TTS_Base`):**
  * Sequence dimension: $d_{\text{model}} = 1024$
  * Depth: 22 Transformer layers
  * Attention Heads: 16 ($d_{\text{head}} = 64$)
  * Position Embeddings: Rotary Position Embeddings (RoPE)
  * Text Dimension: 512, integrated via ConvNeXt V2 blocks
  * Dual Timestep Embedding: Timestep MLP for primary continuous time $t$, and secondary time embedding `time_embed2` (zero-initialized) for student multi-step guidance.

```python
# Extracted from src/f5_tts/model/backbones/dit.py
class DiT(nn.Module):
    def __init__(self, dim=1024, depth=22, heads=16, dim_head=64,
                 dropout=0.1, ff_mult=2, mel_dim=100, text_num_embeds=256,
                 text_dim=512, conv_layers=4, second_time=False):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.time_embed2 = TimestepEmbedding(dim) if second_time else None
        
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
```

### 4.2 UniModel (`src/unimodel.py`)
`UniModel` is the unified container holding both the **Teacher** model (`real_unet`) and the trainable **Student** generator (`feedforward_model`).
* At initialization, `UniModel` deep-copies the teacher DiT weights into:
  1. `self.guidance_model.real_unet`: Frozen teacher DiT providing true vector field targets.
  2. `self.guidance_model.fake_unet`: Critic tracking student generated states.
  3. `self.feedforward_model`: The few-step student generator being optimized.
* In `infer.py`, the two models are unpacked:
  ```python
  self.teacher = self.model.guidance_model.real_unet.to(self.device)
  self.generator = self.model.feedforward_model.to(self.device)
  ```

---

# 5. Duration Prediction & Reinforcement Learning

### 5.1 Architecture: `SpeechLengthPredictor` (`src/duration_predictor.py`)
In non-autoregressive speech synthesis, generation length must be determined prior to solving the ODE. `SpeechLengthPredictor` (SLP) acts as an acoustic-textual cross-attention length predictor:

```python
# Extracted from src/duration_predictor.py
class SpeechLengthPredictor(nn.Module):
    def __init__(self, vocab_size=2545, n_mel=100, hidden_dim=256,
                 n_text_layer=4, n_cross_layer=4, n_head=8, output_dim=301):
        super().__init__()
        # 1. Text Encoder (4 layers)
        self.text_embedder = nn.Embedding(vocab_size+1, hidden_dim, padding_idx=vocab_size)
        self.text_pe = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_text_layer)
        
        # 2. Mel Spectrogram Embedder
        self.mel_embedder = nn.Linear(n_mel, hidden_dim)
        self.mel_pe = PositionalEncoding(hidden_dim)
        
        # 3. Cross-Attention Decoder (4 layers with Causal Mask)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_cross_layer)
        
        # 4. Classification Head (Classifies into 0..300 buckets representing length / 10)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_ids, mel):
        text_features = self.text_encoder(self.text_pe(self.text_embedder(text_ids)))
        mel_features = self.mel_pe(self.mel_embedder(mel))
        seq_len = mel_features.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(mel.device)
        decoder_out = self.decoder(mel_features, text_features, tgt_mask=causal_mask)
        return self.predictor(decoder_out)  # Shape: [B, L_mel, 301]
```

### 5.2 Reinforcement Learning with GRPO (`src/grpo_duration_trainer.py`)
Supervised duration training (predicting ground truth frame counts via L1 or cross-entropy) does not directly optimize perceptual audio intelligibility. DMOSpeech 2 fine-tunes the SLP using **Group Relative Preference Optimization (GRPO)**:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_\theta(d_i|X)}{\pi_{\text{old}}(d_i|X)} \hat{A}_i, \, \text{clip}\left(\frac{\pi_\theta(d_i|X)}{\pi_{\text{old}}(d_i|X)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

1. For each text prompt, $G$ candidate durations $\{d_1, d_2, \dots, d_G\}$ are sampled using Gumbel-Softmax ($\tau = 0.7$).
2. Full audio is synthesized for all $G$ candidates.
3. Multi-metric rewards are computed:
   * **ASR Intelligibility Reward ($R_{\text{WER}}$):** Negative Word Error Rate obtained from Conformer-CTC.
   * **Speaker Similarity Reward ($R_{\text{SIM}}$):** Cosine similarity of ECAPA-TDNN speaker embeddings between reference and generated speech.
4. Relative advantage $\hat{A}_i$ is computed within the group:
   $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\}) + \epsilon}$$

### 5.3 The 1.5k Reward-Hacking Boundary Wall
The repository's default checkpoint for SLP is `model_1500.pt`. Training beyond 1,500 steps encounters severe reward hacking:
* **The ASR Exploitation:** ASR decoders (CTC) struggle with fast, compressed speech but easily parse elongated speech with wide silence gaps. The policy discovers that artificially inflating duration ($T \to T + \Delta T$) drives WER to near-zero.
* **Resulting Audio Failure:** After step 1,500, speech becomes dragging, unnaturally slow, and robotic. Hence, `model_1500.pt` is the Pareto-optimal checkpoint, and duration errors must be mitigated at inference time rather than prolonged RL training.

---

# 6. Distribution Matching Distillation

The student generator is trained using **Distribution Matching Distillation (DMD)** in `src/dmd_trainer.py` to match the marginal probability path of the 32-step teacher in only 4 steps:

1. **DMD Loss:**
   $$\mu_{\text{fake}}(x_t, t) = \text{real\_unet}(x_t, t) - \text{fake\_unet}(x_t, t)$$
   $$\mathcal{L}_{\text{DMD}} = \mathbb{E} \left[ \frac{1}{2} \| x_1^{\text{student}} - (x_t + (1 - t) \mu_{\text{fake}}) \|_2^2 \right]$$
2. **Adversarial Conformer Discriminator Loss (`src/discriminator_conformer.py`):**
   Multi-layer Conformer discriminator with GroupNorm and depthwise separable convolutions enforcing sharp spectral boundaries.
3. **Phonetic CTC Alignment Loss (`src/ctcmodel.py`):**
   Ensures that few-step generation does not drop syllables or collapse phoneme timing.
4. **Speaker Similarity Loss (`src/ecapa_tdnn.py`):**
   Minimizes cosine distance between reference speaker vector and student output vector.

---

# 7. Inference Engine Breakdown (`infer.py`) & The Two Critical Bugs

The complete inference routine is governed by `DMOInference` in `src/infer.py`.

```
               Vanilla DMOSpeech 2 Inference Pipeline
               
  t = 0.0                      t = 0.07                               t = 1.0
  [Noise y_0] ─── Euler ODE ───► [x_stop]
                 (Teacher)           │
                                     │  Hard single-step extrapolation:
                                     │  test_out = x_stop + (1 - t) * pred
                                     ▼
                                [Handover]  <─── C0 DISCONTINUITY & COLLISION!
                                     │
                             Student Refinement
                           4 Steps (swayed Euler)
                                     ▼
                                [Clean Mel]
```

### Offending Code Section 1: The $\mathcal{C}^0$ Boundary Collision
In `_teacher_sampling()` (lines 515–528):
```python
# Setup time steps
t = torch.linspace(0, 1, teacher_steps + 1, device=device, dtype=step_cond.dtype)
t = t[:(t > teacher_stopping_time).float().argmax() + 2]
t = t[:-1]

# Solve ODE with Teacher
trajectory = odeint(fn, y0, t, method="euler")

if teacher_stopping_time < 1.0:
    # HARD DISCONTINUOUS EXTRAPOLATION
    pred = fn(t[-1], trajectory[-1])
    test_out = trajectory[-1] + (1 - t[-1]) * pred
    return test_out
```
* **Failure Analysis:** At $t_{\text{stop}} = 0.07$, the ODE abruptly stops. The function performs a crude linear forward extrapolation: `test_out = trajectory[-1] + (1 - t[-1]) * pred`. Then `_student_sampling()` receives `test_out` and applies:
  ```python
  x0 = torch.randn_like(x1)
  phi = (1 - t) * x0 + t * x1
  pred = self.generator(x=phi, ...)
  output = phi + (1 - t) * pred
  ```
  The student's velocity field $v_{\text{student}}$ and the teacher's velocity field $v_{\text{teacher}}$ are discontinuous at the boundary:
  $$\lim_{t \to t_{\text{stop}}^-} v(x_t, t) \neq \lim_{t \to t_{\text{stop}}^+} v(x_t, t)$$
  This produces high-frequency energy spikes, phase smearing, and harsh timbre shifts at the transition point.

### Offending Code Section 2: Zero Cadence Feedback
In `generate()` and `predict_duration()`:
```python
# predict_duration() calculates static duration
predictions = self.SLP(text_ids=text_ids, mel=pmt_mel)
est_label = predictions.argmax(-1)[..., -1].item() * 10
# ...
# generate() runs ODE blindly on fixed duration
cond, text, step_cond, cond_mask, max_duration = self._prepare_inputs(audio, final_text_list, duration)
```
* **Failure Analysis:** The duration prediction is completely decoupled from the ODE solver. If the SLP predicts 350 frames when the natural phoneme duration requires 290 frames, the ODE transports flow over 350 frames without compensation, dilating formants and creating dragging, robotic speech.

---

# 8. The TAGS-Flow Mathematical Solution & Theoretical Proofs

TAGS-Flow replaces the disjoint `_teacher_sampling()` and `_student_sampling()` with a single unified, mathematically continuous ODE solver: `_tags_flow_sampling()`.

```
                    TAGS-Flow Continuous Blending Manifold
                    
  0.0                                0.20              0.40                               1.0
   │───────────────────────────────────┬─────────────────┬─────────────────────────────────│
   │  Teacher Dominated                │   Cosine Blend  │       Student Dominated         │
   │  w(t) = 0.0                       │   0 < w(t) < 1  │       w(t) = 1.0                │
   │  v_hybrid = v_teacher             │   Smooth Handover│       v_hybrid = v_student      │
   └───────────────────────────────────┴─────────────────┴─────────────────────────────────┘
                                      ▲
                           Continuous First Derivative:
                           dw/dt = 0 at t=0.20 and t=0.40
```

### Theorem 1: $\mathcal{C}^1$-Continuity of the Cosine Blend Manifold
Let the transition window be defined on the compact interval $t \in [t_1, t_2]$. We define the transition weight $w(t): [0, 1] \to [0, 1]$ as:
$$w(t) = \begin{cases} 
0, & t < t_1 \\
\frac{1}{2} \left[ 1 - \cos\left(\pi \frac{t - t_1}{t_2 - t_1}\right) \right], & t_1 \le t \le t_2 \\
1, & t > t_2 
\end{cases}$$

The hybrid velocity field is:
$$v_{\text{hybrid}}(x_t, t) = (1 - w(t)) \cdot v_{\text{teacher}}(x_t, t) + w(t) \cdot v_{\text{student}}(x_t, t)$$

**Proof of Boundary Smoothness:**  
Differentiating $w(t)$ with respect to $t$:
$$\frac{dw}{dt} = \frac{\pi}{2(t_2 - t_1)} \sin\left(\pi \frac{t - t_1}{t_2 - t_1}\right)$$
Evaluating at boundary $t = t_1$:
$$\left. \frac{dw}{dt} \right|_{t = t_1} = \frac{\pi}{2(t_2 - t_1)} \sin(0) = 0$$
Evaluating at boundary $t = t_2$:
$$\left. \frac{dw}{dt} \right|_{t = t_2} = \frac{\pi}{2(t_2 - t_1)} \sin(\pi) = 0$$
Because the time derivative vanishes at both endpoints ($\left.\frac{dw}{dt}\right|_{t_1} = \left.\frac{dw}{dt}\right|_{t_2} = 0$), the handover exhibits **$\mathcal{C}^1$ continuity**. Velocity vector acceleration remains finite, eliminating phase boundary shocks.

---

### Theorem 2: Cadence-Scaling Formulation ($\alpha$)
Let the reference audio mel prompt have duration $T_{\text{ref}}$ (frames) and character token count $N_{\text{ref}}$.  
The ground-truth speaker cadence is:
$$\text{Cadence}_{\text{ref}} = \frac{T_{\text{ref}}}{N_{\text{ref}}} \quad (\text{frames per token})$$

Let the predicted duration for target text (length $N_{\text{target}}$) be $T_{\text{pred}}$:
$$\text{Cadence}_{\text{pred}} = \frac{T_{\text{pred}}}{N_{\text{target}}}$$

The raw cadence ratio is:
$$\alpha_{\text{raw}} = \frac{\text{Cadence}_{\text{ref}}}{\text{Cadence}_{\text{pred}}}$$

To prevent acoustic collapse under extreme duration outlier predictions, $\alpha$ is constrained within a symmetric stability margin $\delta \in [0.10, 0.20]$ (default $\delta = 0.15$):
$$\alpha = \text{clamp}\left( \frac{\text{Cadence}_{\text{ref}}}{\text{Cadence}_{\text{pred}}}, \, 1 - \delta, \, 1 + \delta \right)$$

**Integration into Euler Step:**
$$x_{t+h} = x_t + h \cdot \big( \alpha \cdot v_{\text{hybrid}}(x_t, t) \big)$$

* If the SLP over-estimates duration ($T_{\text{pred}}$ too large $\implies \text{Cadence}_{\text{pred}}$ too large), then $\alpha < 1.0$. Velocity is throttled so the trajectory does not traverse empty space and create stretched vowels.
* If the SLP under-estimates duration ($T_{\text{pred}}$ too small $\implies \alpha > 1.0$), velocity is boosted so phonetic transitions complete before the frame budget ends.

---

# 9. TAGS-Flow Implementation Details

Implementation requires updating **`src/infer.py`** without touching pre-trained weights or external training files.

### 9.1 Helper: Velocity Predictors
```python
def _teacher_velocity(self, x, cond, text, t, cfg_strength=1.0):
    """Evaluates the teacher DiT velocity field."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = self.teacher(
            x=x, cond=cond, text=text, time=t, mask=None,
            drop_audio_cond=False, drop_text=False
        )
        if cfg_strength > 1e-5:
            null_pred = self.teacher(
                x=x, cond=cond, text=text, time=t, mask=None,
                drop_audio_cond=True, drop_text=True
            )
            pred = pred + (pred - null_pred) * cfg_strength
    return pred

def _student_velocity(self, x, cond, text, t):
    """Evaluates the student generator velocity field."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = self.generator(
            x=x, cond=cond, text=text, time=t,
            drop_audio_cond=False, drop_text=False
        )
    return pred
```

### 9.2 The TAGS-Flow ODE Solver: `_tags_flow_sampling()`
```python
def _tags_flow_sampling(
    self, step_cond, text, cond_mask, max_duration, duration,
    steps=16, t_start_blend=0.20, t_end_blend=0.40,
    alpha=1.0, cfg_strength=1.0, sway_coeff=-1.0
):
    """
    Unified TAGS-Flow ODE integration with C1 cosine blending and cadence-scaling.
    """
    device = step_cond.device
    
    # 1. Initialize prior noise y_0 ~ N(0, I)
    y0 = []
    for dur in duration:
        y0.append(torch.randn(dur, 100, device=device, dtype=step_cond.dtype))
    x = pad_sequence(y0, padding_value=0, batch_first=True)
    
    # 2. Time discretization grid with Sway coefficient
    t_steps = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=step_cond.dtype)
    if sway_coeff is not None and sway_coeff != 0:
        t_steps = t_steps + sway_coeff * (torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)
        
    # 3. ODE Integration Loop
    for i in range(steps):
        t_curr = t_steps[i]
        t_next = t_steps[i + 1]
        h = t_next - t_curr
        t_val = t_curr.item()
        
        # Compute smooth cosine weight w(t)
        if t_val < t_start_blend:
            w = 0.0
        elif t_val > t_end_blend:
            w = 1.0
        else:
            tau = (t_val - t_start_blend) / (t_end_blend - t_start_blend)
            w = 0.5 * (1.0 - math.cos(math.pi * tau))
            
        t_tensor = t_curr.unsqueeze(0)
        
        # Evaluate vector fields conditionally
        if w == 0.0:
            v_hybrid = self._teacher_velocity(x, step_cond, text, t_tensor, cfg_strength)
        elif w == 1.0:
            v_hybrid = self._student_velocity(x, step_cond, text, t_tensor)
        else:
            v_t = self._teacher_velocity(x, step_cond, text, t_tensor, cfg_strength)
            v_s = self._student_velocity(x, step_cond, text, t_tensor)
            v_hybrid = (1.0 - w) * v_t + w * v_s
            
        # Apply Cadence Scaling α and advance Euler step
        x = x + h * (alpha * v_hybrid)
        
    return x
```

### 9.3 Updated `generate()` in `infer.py`
```python
# Compute cadence ratio inside generate():
ref_audio_frames = audio.shape[-1] // self.hop_length
ref_token_count = max(len(ref_text), 1)
target_token_count = max(len(gen_text), 1)

cadence_ref = ref_audio_frames / ref_token_count
cadence_pred = (duration - ref_audio_frames) / target_token_count

raw_alpha = cadence_ref / max(cadence_pred, 1e-6)
delta = 0.15
alpha = max(1.0 - delta, min(1.0 + delta, raw_alpha))

# Execute TAGS-Flow ODE
x1 = self._tags_flow_sampling(
    step_cond=step_cond,
    text=text,
    cond_mask=cond_mask,
    max_duration=max_duration,
    duration=duration,
    steps=16,
    t_start_blend=0.20,
    t_end_blend=0.40,
    alpha=alpha,
    cfg_strength=cfg_strength,
    sway_coeff=sway_coefficient
)

# Decode via Vocos Vocoder
mel = x1.permute(0, 2, 1) * self.scale
generated_wave = self.vocos.decode(mel[..., cond_mask.sum():])
return generated_wave
```

---

# 10. Ablation Study, Benchmarking & Experimental Verification

### 10.1 The 4-Way Ablation Design
To validate both theoretical pillars independently, evaluate across LibriSpeech-PC 4-second zero-shot evaluation pairs:

| Experiment | Handover Mode | Velocity Blending $w(t)$ | Cadence Scaler $\alpha$ | Theoretical Target |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (DMOSpeech 2)** | Hard cutoff ($t=0.07$) | None ($\mathcal{C}^0$ Discontinuous) | $\alpha = 1.0$ (Unscaled) | Replicates official repo behavior |
| **Ablation A** | Hard cutoff ($t=0.07$) | None ($\mathcal{C}^0$ Discontinuous) | $\alpha = \text{clamp}(\dots)$ | Tests cadence compensation alone |
| **Ablation B** | Continuous ($[0.2, 0.4]$) | $\mathcal{C}^1$ Cosine Manifold | $\alpha = 1.0$ (Unscaled) | Tests boundary smoothing alone |
| **Full TAGS-Flow** | Continuous ($[0.2, 0.4]$) | $\mathcal{C}^1$ Cosine Manifold | $\alpha = \text{clamp}(\dots)$ | **Proposed Complete System** |

### 10.2 Metric Battery
1. **Intelligibility — Word Error Rate (WER %):**
   Evaluated using Whisper-Large-v3 on generated speech.
2. **Speaker Similarity (SIM-O & SIM-R):**
   Cosine similarity of WavLM-TDNN / ECAPA-TDNN embeddings between prompt and output.
3. **Perceptual Acoustic Quality (UTMOS):**
   Deep neural MOS estimator predicting naturalness and phase distortion.
4. **Computational Efficiency — Real-Time Factor (RTF):**
   $$\text{RTF} = \frac{\text{Wall Clock Generation Time (seconds)}}{\text{Generated Audio Duration (seconds)}}$$

---

# 11. Master Hyperparameter Reference Table

| Hyperparameter | Symbol | Value in Codebase | Description / Role |
| :--- | :---: | :---: | :--- |
| **Sampling Rate** | $f_s$ | `24000 Hz` | Native acoustic sampling frequency |
| **Mel Spectrogram Dimension** | $D_{\text{mel}}$ | `100` | Number of mel filterbank channels |
| **FFT Window Length** | $N_{\text{FFT}}$ | `1024` | STFT analysis frame length |
| **Hop Length** | $H$ | `256` | STFT hop length ($256 / 24000 = 10.67\text{ ms/frame}$) |
| **DiT Model Dimension** | $d_{\text{model}}$ | `1024` | Hidden dimension of Transformer backbone |
| **DiT Number of Layers** | $L$ | `22` | Number of DiT Blocks in `F5TTS_Base` |
| **DiT Heads** | $H$ | `16` | Multi-head attention count ($d_{\text{head}} = 64$) |
| **Duration Classes** | $C_{\text{dur}}$ | `301` | Number of length buckets ($0 \dots 3000$ frames in steps of 10) |
| **SLP Hidden Dimension** | $d_{\text{slp}}$ | `256` | SpeechLengthPredictor representation dimension |
| **SLP Text Layers** | $L_{\text{text}}$ | `4` | Self-attention layers for text conditioning |
| **SLP Cross Layers** | $L_{\text{cross}}$ | `4` | Cross-attention layers decoding mel into duration logits |
| **GRPO Group Size** | $G$ | `8` | Candidate samples drawn per prompt |
| **GRPO KL Penalty** | $\beta$ | `0.04` | Divergence penalty against reference duration model |
| **Optimal SLP Checkpoint** | — | `model_1500.pt` | Checkpoint prior to ASR reward-hacking drift |
| **Teacher Stop Time (Old)** | $t_{\text{stop}}$ | `0.07` (or `0.30`) | Abrupt teacher exit threshold in DMOSpeech 2 |
| **TAGS Blend Start** | $t_1$ | `0.20` | Lower bound of continuous cosine handover |
| **TAGS Blend End** | $t_2$ | `0.40` | Upper bound of continuous cosine handover |
| **Cadence Clamping Bound** | $\delta$ | `0.15` | Constrains $\alpha \in [0.85, 1.15]$ for acoustic stability |
| **Classifier-Free Guidance** | $\gamma_{\text{CFG}}$ | `1.0` | Guidance weight applied to unconditional null token |
| **Sway Coefficient** | $s$ | `-1.0` | Warps numerical time steps towards higher noise regions |
