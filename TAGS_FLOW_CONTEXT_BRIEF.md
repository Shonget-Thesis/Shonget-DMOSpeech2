# 🧠 COMPLETE CONTEXT BRIEF: TAGS-Flow Thesis
## For AI Model Onboarding — Read This First Before Any Task

**Project:** DMOSpeech 2 → TAGS-Flow Extension  
**Codebase Root:** `Shonget-DMOSpeech2/`  
**Primary File to Modify:** `src/infer.py`  
**Nature of Work:** Training-free inference algorithm replacement. No model retraining required.

---

## PART 1: What This Project Is

**DMOSpeech 2** is a **zero-shot text-to-speech (TTS)** system. Given:
- A **reference audio clip** (e.g., 5 seconds of a speaker's voice), and
- A **target text string** (what you want the speaker to say),

it synthesizes new speech in that speaker's voice — without ever having trained on that speaker.

The paper is: *"DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis"* (Li et al., July 2025). It builds on **F5-TTS** (Chen et al., Oct 2024).

The thesis being written proposes **TAGS-Flow**, a modification to the inference algorithm of DMOSpeech 2 that is mathematically smoother and more accurate.

---

## PART 2: Repository File Structure

```
Shonget-DMOSpeech2/
├── README.md                          # Setup & usage instructions
├── requirements.txt                   # Python dependencies (torch, torchaudio, torchdiffeq, etc.)
├── src/
│   ├── infer.py                       ★ THE MAIN FILE — all inference logic lives here
│   ├── duration_predictor.py          # SpeechLengthPredictor model definition
│   ├── unimodel.py                    # UniModel: wraps teacher + student + guidance
│   ├── guidance_model.py              # Guidance class: teacher/student loss computation
│   ├── dmd_trainer.py                 # DMD training loop (not relevant for inference)
│   ├── grpo_duration_trainer.py       # GRPO RL training for duration predictor
│   ├── duration_trainer.py            # Pre-training for duration predictor
│   ├── demo.ipynb                     # Jupyter notebook showing how to call DMOInference
│   └── f5_tts/                        # F5-TTS submodule (teacher model architecture, tokenizer, vocoder utils)
│       ├── model/                     # DiT, UNetT, CFM, MelSpec, etc.
│       └── infer/                     # load_vocoder, preprocess_ref_audio_text, etc.
├── ckpts/                             # (user downloads) model checkpoints
│   ├── model_85000.pt                 # Main DMOSpeech2 checkpoint (teacher + student weights)
│   └── model_1500.pt                  # GRPO-finetuned duration predictor checkpoint
└── THESIS_PATH_*.md                   # Thesis planning documents
```

---

## PART 3: The Full Inference Pipeline (How Speech is Generated)

The entry point is `DMOInference` in `src/infer.py`. Here is the exact call pattern from `demo.ipynb`:

```python
from infer import DMOInference

# 1. Initialize (loads models into GPU memory)
tts = DMOInference(
    student_checkpoint_path="../ckpts/model_85000.pt",
    duration_predictor_path="../ckpts/model_1500.pt",
    device="cuda",
    model_type="F5TTS_Base"
)

# 2. Generate speech
generated_audio = tts.generate(
    gen_text="I don't really care what you call me.",
    audio_path="path/to/reference_speaker.wav",
    prompt_text="Some call me nature, others call me mother nature.",
    teacher_steps=16,
    teacher_stopping_time=0.07,   # KEY PARAMETER: where teacher hands off to student
    student_start_step=1,
    temperature=0,                # 0 = deterministic duration prediction
    eta=1.0,                      # stochasticity (0=DDIM, 1=DDPM)
    cfg_strength=2.0,
)
# generated_audio is a numpy array at 24000 Hz sample rate
```

The pipeline has **4 sequential stages**:

```
Reference Audio + Target Text
        |
        v
Stage 1: Duration Predict       <- SpeechLengthPredictor (GRPO-trained, model_1500.pt)
"How many mel frames?"             Answers: e.g., 320 frames = ~3.4 seconds of audio
        |
        v
Stage 2: Teacher Sampling       <- F5-TTS DiT model (real_unet from model_85000.pt)
ODE from noise to rough mel        Runs 16 Euler steps, stops at t=0.07 (early exit!)
        |
        v
Stage 3: Student Sampling       <- Distilled DiT (feedforward_model from model_85000.pt)
4-step fast refinement             Takes teacher's rough output, polishes it fast
        |
        v
Stage 4: Vocoder (Vocos)        <- Converts mel spectrogram to audio waveform
```

---

## PART 4: Key Models and Their Roles

### 4.1 SpeechLengthPredictor (`src/duration_predictor.py`)

The neural network that predicts how long the generated audio should be. It is a Transformer encoder-decoder:

```python
class SpeechLengthPredictor(nn.Module):
    def __init__(self, vocab_size=2545, n_mel=100, hidden_dim=512,
                 n_text_layer=4, n_cross_layer=4, n_head=8, output_dim=301):
        # Text Encoder: embeds + positional-encodes the (prompt_text + target_text) tokens
        self.text_embedder = nn.Embedding(vocab_size+1, hidden_dim, padding_idx=vocab_size)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_text_layer)
        
        # Mel Encoder: projects reference audio mel frames to hidden_dim
        self.mel_embedder = nn.Linear(n_mel, hidden_dim)
        
        # Cross-Attention Decoder: attends mel over text to predict length
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_cross_layer)
        
        # Output: 301 classes (0-300, each represents x10 mel frames)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_ids, mel):
        # text_ids: tokenized [prompt_text + " " + target_text]
        # mel: mel spectrogram of reference audio, shape (B, L_mel, 100)
        text_features = self.text_encoder(self.text_pe(self.text_embedder(text_ids)))
        mel_features = self.mel_pe(self.mel_embedder(mel))
        decoder_out = self.decoder(mel_features, text_features, tgt_mask=causal_mask)
        return self.predictor(decoder_out).squeeze(-1)  # shape: (B, L_mel, 301)
```

**How prediction is extracted** (from `infer.py predict_duration`):
```python
with torch.no_grad():
    predictions = self.SLP(text_ids=text_ids, mel=pmt_mel)

# Only look at the last mel frame's prediction
predictions = predictions[:, -1, :]  # shape: (B, 301)

# Clamp to a range around the rate-based estimate
predictions[:, :min_duration] = float('-inf')
predictions[:, max_duration:] = float('-inf')

# Argmax (temperature=0) or sample (temperature>0)
if temperature == 0:
    est_label = predictions.argmax(-1).item() * 10  # -> mel frames
else:
    probs = torch.softmax(predictions / temperature, dim=-1)
    est_label = torch.multinomial(probs.squeeze(0), 1).item() * 10
```

**This model was fine-tuned with GRPO** (Group Relative Policy Optimization) — reinforcement learning using WER and speaker similarity as reward signals. That is the thesis's existing contribution from DMOSpeech 2.

---

### 4.2 UniModel (`src/unimodel.py`)

`UniModel` is a container that holds three DiT (Diffusion Transformer) models loaded from `model_85000.pt`:

```python
class UniModel(nn.Module):
    def __init__(self, model: DiT, ...):
        real_unet = copy.deepcopy(model)   # THE TEACHER (F5-TTS quality)
        fake_unet = copy.deepcopy(model)   # discriminator (used during training only)
        
        self.guidance_model = Guidance(
            real_unet=real_unet,           # teacher lives here
            fake_unet=fake_unet,
        )
        self.feedforward_model = copy.deepcopy(model)  # THE STUDENT (fast, distilled)
```

During inference, only two of these are used:
- `self.model.guidance_model.real_unet` — **teacher** (high-quality, slow)
- `self.model.feedforward_model` — **student** (fast, 4 steps)

```python
# From infer.py _setup_models():
self.generator = self.model.feedforward_model.to(self.device)    # student
self.teacher   = self.model.guidance_model.real_unet.to(self.device)  # teacher
```

Both are `DiT` (Diffusion Transformer) models — 22-layer transformers with 16 attention heads, operating on mel spectrograms.

---

### 4.3 The DiT Model (from F5-TTS)

The DiT (`f5_tts/model/`) takes:
- `x`: current noisy mel spectrogram (B, T, 100)
- `cond`: conditioning (reference audio mel, masked)
- `text`: tokenized text
- `time`: current ODE timestep t in [0, 1]

And outputs:
- A **velocity vector** `v(x, t)` — the direction to move in mel-spectrogram space to get from noise (t=0) to clean speech (t=1)

This is the core of **Continuous Normalizing Flow / Flow Matching**.

---

## PART 5: The Current Inference Code (What Exists Now)

### 5.1 `generate()` — The Main Orchestrator

```python
def generate(self, gen_text, audio_path, prompt_text=None,
             teacher_steps=16, teacher_stopping_time=0.07,
             student_start_step=1, duration=None,
             dp_softmax_range=0.7, temperature=0,
             eta=1.0, cfg_strength=2.0, sway_coefficient=-1.0, verbose=False):

    # Step 1: ASR to get prompt text if not provided
    if prompt_text is None:
        prompt_text = transcribe(audio_path)

    # Step 2: Predict duration (how many mel frames to generate)
    if duration is None:
        duration = self.predict_duration(audio_path, gen_text, prompt_text,
                                         dp_softmax_range, temperature)
    # duration is now an integer: e.g., 320 mel frames

    # Step 3: Load & normalize reference audio, tokenize text...
    # [normalization, resampling to 24000 Hz, pinyin conversion]

    # Step 4: Calculate total sequence length
    ref_audio_len = audio.shape[-1] // self.hop_length  # reference frames
    duration = ref_audio_len + duration                  # total = ref + generated

    # Step 5: Run inference (NO GRADIENTS)
    with torch.inference_mode():
        cond, text, step_cond, cond_mask, max_duration, duration_tensor = \
            self._prepare_inputs(audio, final_text_list, duration)

        # STAGE 2: Teacher Sampling
        if teacher_steps > 0 and student_start_step > 0:
            x1 = self._teacher_sampling(
                step_cond, text, cond_mask, max_duration, duration_tensor,
                teacher_steps, teacher_stopping_time,
                eta, cfg_strength, verbose, sway_coefficient
            )
        else:
            x1 = step_cond  # pure student mode

        # STAGE 3: Student Sampling
        x1 = self._student_sampling(x1, cond, text, student_start_step,
                                     verbose, sway_coefficient)

        # STAGE 4: Vocoder
        mel = x1.permute(0, 2, 1) * self.scale
        generated_wave = self.vocos.decode(mel[..., cond_mask.sum():])
        # cond_mask.sum() strips the reference portion, leaving only generated audio

    return generated_wave.cpu().numpy().squeeze()
```

---

### 5.2 `_teacher_sampling()` — CURRENT IMPLEMENTATION (PROBLEM IS HERE)

```python
def _teacher_sampling(self, step_cond, text, cond_mask, max_duration, duration,
                      teacher_steps, teacher_stopping_time, eta, cfg_strength,
                      verbose, sway_sampling_coef=-1):

    # Pre-generate noise for stochastic sampling (DDPM-style, controlled by eta)
    noise_seq = None
    if eta > 0:
        noise_seq = [torch.randn(1, max_duration, 100, device=device)
                     for _ in range(teacher_steps)]

    def fn(t, x):
        # Predict velocity vector from teacher DiT
        pred = self.teacher(
            x=x, cond=step_cond, text=text, time=t, mask=None,
            drop_audio_cond=False, drop_text=False
        )
        # Classifier-Free Guidance: run conditioned + unconditioned, then extrapolate
        if cfg_strength > 1e-5:
            null_pred = self.teacher(
                x=x, cond=step_cond, text=text, time=t, mask=None,
                drop_audio_cond=True, drop_text=True
            )
            pred = pred + (pred - null_pred) * cfg_strength

        # Add DDPM stochasticity
        if eta > 0 and noise_seq is not None:
            alpha_t = 1.0 - t.item()
            sigma_t = t.item()
            noise_scale = torch.sqrt(torch.tensor(
                (sigma_t**2) / (alpha_t**2 + sigma_t**2) * eta, device=device
            ))
            return pred + noise_scale * noise_seq[step_idx]
        return pred

    # Initialize from pure Gaussian noise
    y0 = pad_sequence([torch.randn(dur, 100, device=device) for dur in duration],
                      padding_value=0, batch_first=True)

    # Build time grid with optional sway sampling
    t = torch.linspace(0, 1, teacher_steps + 1, device=device)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    # TRUNCATE at teacher_stopping_time — ABRUPT CUTOFF (the bug TAGS-Flow fixes)
    t = t[:(t > teacher_stopping_time).float().argmax() + 2]
    t = t[:-1]

    # Solve ODE using Euler method
    trajectory = odeint(fn, y0, t, method="euler")

    # HARD HANDOFF — C0 discontinuity
    if teacher_stopping_time < 1.0:
        pred = fn(t[-1], trajectory[-1])
        test_out = trajectory[-1] + (1 - t[-1]) * pred  # single extrapolation jump
        return test_out   # student will take over from here with misaligned velocity
    else:
        return trajectory[-1]
```

**The problem:** At `teacher_stopping_time` (default `0.07`), the teacher stops and does one final extrapolation. The student then starts from this point, but its velocity field was never aligned with the teacher's at the boundary — causing a velocity vector collision (C0 discontinuity).

---

### 5.3 `_student_sampling()` — CURRENT IMPLEMENTATION

```python
def _student_sampling(self, x1, cond, text, student_start_step, verbose, sway_coeff=-1):
    # Fixed 4-step grid: [0, 0.25, 0.5, 0.75]
    steps = torch.Tensor([0, 0.25, 0.5, 0.75])
    steps = steps + sway_coeff * (torch.cos(torch.pi / 2 * steps) - 1 + steps)
    steps = steps[student_start_step:]   # skip steps already covered by teacher

    for step in steps:
        time = torch.Tensor([step]).to(x1.device)
        x0 = torch.randn_like(x1)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1       # interpolate between noise and current estimate

        with torch.no_grad():
            pred = self.generator(
                x=phi, cond=cond, text=text, time=time,
                drop_audio_cond=False, drop_text=False
            )
            output = phi + (1 - t) * pred  # flow matching update

        x1 = output

    return x1
```

**The problem:** The student's 4 fixed steps have no awareness of whether the predicted duration was accurate. If the duration predictor drifted, there is no velocity compensation.

---

### 5.4 `predict_duration()` — CURRENT IMPLEMENTATION

```python
def predict_duration(self, pmt_wav_path, tar_text, pmt_text,
                     dp_softmax_range=0.7, temperature=0):

    # Load reference audio and convert to mel
    pmt_wav, sr = torchaudio.load(pmt_wav_path)
    pmt_mel = self.wav2mel(pmt_wav).permute(0, 2, 1)   # shape: (1, L_mel, 100)

    # Tokenize both texts (pinyin for Chinese/English)
    tar_tokens = self._convert_to_pinyin(list(tar_text))
    pmt_tokens = self._convert_to_pinyin(list(pmt_text))

    # Rate-based anchor: scale reference duration by text length ratio
    ref_text_len = len(pmt_tokens)
    gen_text_len  = len(tar_tokens)
    ref_audio_len = pmt_mel.size(1)
    duration = int(ref_audio_len / ref_text_len * gen_text_len / speed)
    duration = duration // 10   # convert to class index space (divide by 10)

    # Clip range for neural prediction
    min_duration = max(int(duration * dp_softmax_range), 0)
    max_duration = min(int(duration * (1 + dp_softmax_range)), 301)

    # Concatenate tokens: [prompt_tokens] + [space] + [target_tokens]
    all_tokens = pmt_tokens + [' '] + tar_tokens
    text_ids = list_str_to_idx([all_tokens], self.vocab_char_map).to(self.device)
    text_ids = text_ids.masked_fill(text_ids == -1, self.vocab_size)

    # Neural prediction
    with torch.no_grad():
        predictions = self.SLP(text_ids=text_ids, mel=pmt_mel)
    predictions = predictions[:, -1, :]             # last mel frame output
    predictions[:, :min_duration] = float('-inf')   # mask below range
    predictions[:, max_duration:] = float('-inf')   # mask above range

    if temperature == 0:
        est_label = predictions.argmax(-1)[..., -1].item() * 10
    else:
        probs = torch.softmax(predictions / temperature, dim=-1)
        est_label = torch.multinomial(probs.squeeze(0), 1).item() * 10

    return est_label  # integer: number of mel frames for the TARGET portion only
```

---

## PART 6: The Two Problems TAGS-Flow Solves

### Problem 1 — The C0 Boundary Discontinuity

**Location in code:** End of `_teacher_sampling()`.

**Evidence:**
```python
# This is the exact offending code:
if teacher_stopping_time < 1.0:
    pred = fn(t[-1], trajectory[-1])
    test_out = trajectory[-1] + (1 - t[-1]) * pred
    return test_out   # ABRUPT handoff — velocity fields misaligned
```

**Why it's wrong mathematically:** The teacher trajectory arrives at state `x(t_stop)` with velocity `v_teacher(x(t_stop), t_stop)`. The student then starts from this state using `v_student(x(t_stop), t_stop)`. These two velocity vectors were not constrained to be equal during distillation — so the ODE trajectory has a sudden direction change (C0 discontinuity).

**TAGS-Flow fix:** Smooth cosine blend over window [t1, t2]:
```
w(t) = 0.5 * (1 - cos(pi * (t - t1) / (t2 - t1)))
v_hybrid = (1 - w(t)) * v_teacher + w(t) * v_student
```
At t=t1: w=0 (all teacher), dw/dt=0 — no jump.
At t=t2: w=1 (all student), dw/dt=0 — no jump.
This gives C1 continuity.

---

### Problem 2 — No Cadence Feedback / Duration Drift

**Location in code:** `odeint(fn, y0, t, method="euler")` in `_teacher_sampling()`.

**Why it's wrong:** Duration is predicted once and locked in. If the predicted duration is 300 frames but the optimal is 280, the ODE must compress spectral energy into 300 frames — causing slurring (underestimate) or trailing silence/hallucination (overestimate).

**TAGS-Flow fix:** Dynamic Velocity Scaler alpha injected into every Euler step:
```python
cadence_ref  = ref_audio_len / ref_text_token_count    # frames per token in reference
cadence_pred = predicted_duration / gen_text_token_count  # frames per token predicted
delta = 0.15  # max clamp range
alpha = clamp(cadence_ref / cadence_pred, 1 - delta, 1 + delta)

# Modified Euler step (instead of x = x + h * v_t):
x = x + h * (alpha * v_hybrid)
```

---

## PART 7: The TAGS-Flow Algorithm — Full Specification

### 7.1 Mathematical Formulation

**The ODE being solved:**
```
dx_t/dt = v_t(x_t, t),   x(0) ~ N(0,I),   x(1) = mel spectrogram
```

**C1-Smooth Transition Weight:**
```
w(t) = 0.5 * [1 - cos(pi * (t - t1) / (t2 - t1))],   t in [t1, t2]
```

**Hybrid Velocity Field:**
```
v_hybrid(x_t, t) = (1 - w(t)) * v_teacher(x_t, t) + w(t) * v_student(x_t, t)
```

**Dynamic Velocity Scaling:**
```
alpha = clamp(cadence_ref / cadence_pred,  1 - delta,  1 + delta)
```

**Modified ODE Step:**
```
x_{t+h} = x_t + h * (alpha * v_hybrid(x_t, t))
```

---

### 7.2 Algorithm Pseudocode (Python-style)

```python
# Algorithm: TAGS-Flow Inference
# Inputs: gen_text, ref_audio, N steps, t1=0.20, t2=0.40, delta=0.15
# Output: mel spectrogram Y

# Stage 1: Duration + Cadence
T_hat   = SpeechLengthPredictor(gen_text, ref_audio)      # predicted mel frames
alpha   = clamp(cadence_ref / (T_hat / len(gen_tokens)), 1-delta, 1+delta)

# Stage 2-3: TAGS-Flow ODE
x       = randn(T_hat, 100)             # Gaussian noise initialization
t_grid  = linspace(0, 1, N+1)          # uniform or sway-adjusted time grid

for k in range(N):
    t = t_grid[k]
    h = t_grid[k+1] - t_grid[k]

    if t < t1:
        # Pure teacher zone
        v_t = v_teacher(x, t) + cfg_guidance + ddpm_noise(eta)

    elif t1 <= t <= t2:
        # Smooth blend zone — BOTH models run
        w_t = 0.5 * (1 - cos(pi * (t - t1) / (t2 - t1)))
        v_T = v_teacher(x, t) + cfg_guidance + ddpm_noise(eta)
        v_S = v_student(x, t)
        v_t = (1 - w_t) * v_T + w_t * v_S

    else:
        # Pure student zone
        v_t = v_student(x, t)

    # Dynamic cadence-scaled Euler step
    x = x + h * (alpha * v_t)

Y = x   # final mel spectrogram -> Vocos -> audio
```

---

## PART 8: Where Exactly to Make Changes in `src/infer.py`

### Change 1: Update `predict_duration()` to also return cadence info

```python
# Add at end of predict_duration(), before return:
cadence_ref  = ref_audio_len / ref_text_len
cadence_pred = est_label / gen_text_len
return est_label, cadence_ref, cadence_pred
```

### Change 2: Compute alpha in `generate()`

```python
# After calling predict_duration():
duration, cadence_ref, cadence_pred = self.predict_duration(...)
delta = 0.15
alpha = max(1 - delta, min(1 + delta, cadence_ref / cadence_pred))
```

### Change 3: Add `_tags_flow_sampling()` method

This new method replaces both `_teacher_sampling()` and `_student_sampling()`:

```python
def _tags_flow_sampling(self, step_cond, cond, text, cond_mask, max_duration,
                         duration, N=6, t1=0.20, t2=0.40,
                         alpha=1.0, eta=1.0, cfg_strength=2.0,
                         sway_coeff=-1.0, verbose=False):
    """
    C1-continuous hybrid teacher-student ODE sampler with dynamic velocity scaling.
    Replaces the separate _teacher_sampling + _student_sampling methods.
    """
    import math
    device = step_cond.device

    # Initialize from Gaussian noise
    y0 = pad_sequence(
        [torch.randn(dur, 100, device=device, dtype=step_cond.dtype) for dur in duration],
        padding_value=0, batch_first=True
    )

    # Pre-generate noise for DDPM stochasticity
    noise_seq = [torch.randn_like(y0) for _ in range(N)] if eta > 0 else None

    # Build time grid
    t_grid = torch.linspace(0, 1, N + 1, device=device, dtype=step_cond.dtype)
    if sway_coeff is not None:
        t_grid = t_grid + sway_coeff * (torch.cos(math.pi / 2 * t_grid) - 1 + t_grid)

    x = y0
    for k in range(N):
        t_val    = t_grid[k].item()
        h        = (t_grid[k+1] - t_grid[k]).item()
        t_tensor = t_grid[k].reshape(1)

        if verbose:
            print(f"TAGS-Flow step {k+1}/{N}, t={t_val:.3f}, zone={'T' if t_val<t1 else 'B' if t_val<=t2 else 'S'}")

        # Zone A: Teacher only
        if t_val < t1:
            v_t = self._teacher_velocity(x, step_cond, text, t_tensor,
                                          cfg_strength, noise_seq, k, eta)
            x = x + h * (alpha * v_t)

        # Zone B: Smooth cosine blend (the novel C1 contribution)
        elif t1 <= t_val <= t2:
            w_t = 0.5 * (1.0 - math.cos(math.pi * (t_val - t1) / (t2 - t1)))
            v_T  = self._teacher_velocity(x, step_cond, text, t_tensor,
                                           cfg_strength, noise_seq, k, eta)
            v_S  = self._student_velocity(x, cond, text, t_tensor)
            v_t  = (1.0 - w_t) * v_T + w_t * v_S
            x    = x + h * (alpha * v_t)

        # Zone C: Student only (uses flow-matching re-noise update)
        else:
            x0  = torch.randn_like(x)
            t_s = t_tensor.unsqueeze(-1).unsqueeze(-1)
            phi = (1 - t_s) * x0 + t_s * x
            with torch.no_grad():
                pred = self.generator(x=phi, cond=cond, text=text, time=t_tensor,
                                       drop_audio_cond=False, drop_text=False)
                x = phi + (1 - t_s) * pred  # standard flow-matching step

    return x


def _teacher_velocity(self, x, step_cond, text, t, cfg_strength, noise_seq, step_idx, eta):
    """Compute teacher velocity with CFG and optional DDPM noise."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = self.teacher(x=x, cond=step_cond, text=text, time=t,
                             mask=None, drop_audio_cond=False, drop_text=False)
        if cfg_strength > 1e-5:
            null_pred = self.teacher(x=x, cond=step_cond, text=text, time=t,
                                      mask=None, drop_audio_cond=True, drop_text=True)
            pred = pred + (pred - null_pred) * cfg_strength
    if eta > 0 and noise_seq is not None:
        alpha_t = 1.0 - t.item()
        sigma_t = t.item()
        if alpha_t**2 + sigma_t**2 > 0:
            noise_scale = torch.sqrt(torch.tensor(
                (sigma_t**2) / (alpha_t**2 + sigma_t**2) * eta, device=x.device
            ))
            pred = pred + noise_scale * noise_seq[step_idx]
    return pred


def _student_velocity(self, x, cond, text, t):
    """Compute student velocity for use in the blend zone."""
    x0  = torch.randn_like(x)
    t_s = t.unsqueeze(-1).unsqueeze(-1)
    phi = (1 - t_s) * x0 + t_s * x
    with torch.no_grad():
        pred = self.generator(x=phi, cond=cond, text=text, time=t,
                               drop_audio_cond=False, drop_text=False)
    return pred
```

### Change 4: Update `generate()` to call `_tags_flow_sampling()`

```python
# BEFORE (two separate calls):
x1 = self._teacher_sampling(step_cond, text, cond_mask, max_duration,
                              duration_tensor, teacher_steps,
                              teacher_stopping_time, eta, cfg_strength,
                              verbose, sway_coefficient)
x1 = self._student_sampling(x1, cond, text, student_start_step, verbose, sway_coefficient)

# AFTER (single unified TAGS-Flow call):
x1 = self._tags_flow_sampling(
    step_cond=step_cond, cond=cond, text=text,
    cond_mask=cond_mask, max_duration=max_duration,
    duration=duration_tensor,
    N=total_steps,        # e.g., 6
    t1=t1,                # transition window start, e.g., 0.20
    t2=t2,                # transition window end, e.g., 0.40
    alpha=alpha,          # computed from cadence ratio
    eta=eta,
    cfg_strength=cfg_strength,
    sway_coeff=sway_coefficient,
    verbose=verbose
)
```

---

## PART 9: What Does NOT Change

The following files and components are completely untouched by TAGS-Flow:

| Component | File | Status |
|---|---|---|
| SpeechLengthPredictor architecture | `duration_predictor.py` | No change |
| UniModel / Guidance class | `unimodel.py`, `guidance_model.py` | No change |
| Model weights | `model_85000.pt`, `model_1500.pt` | No change — training-free |
| Vocoder (Vocos) | loaded via `f5_tts` | No change |
| Mel spectrogram module | `f5_tts/model/modules.py` | No change |
| Tokenizer (pinyin) | `f5_tts/model/utils.py` | No change |
| `DMOInference.__init__()` | `infer.py` | Minor: add `t1`, `t2`, `delta` default params |
| Training code | `dmd_trainer.py`, `grpo_duration_trainer.py` | Not used |

---

## PART 10: Evaluation Plan

After implementing TAGS-Flow, evaluate by running the same 100 test sentences through both systems and comparing:

| Metric | Tool | What it measures |
|---|---|---|
| **WER** (Word Error Rate, lower is better) | OpenAI Whisper ASR | Intelligibility — fewer errors = clearer speech |
| **SIM** (Speaker Similarity, higher is better) | ECAPA-TDNN cosine similarity | Voice cloning accuracy vs. reference speaker |
| **RTF** (Real-Time Factor, lower is better) | `time.time()` / audio duration | Speed — RTF < 1.0 means faster than real-time |
| **UTMOS** (MOS neural score, higher is better) | UTMOS neural scorer | Perceptual naturalness |

The existing `src/ecapa_tdnn.py` provides the speaker similarity model.
RTF is already computed in `demo.ipynb` as `processing_time / audio_duration`.

---

## PART 11: Thesis Chapter to Code Mapping

| Thesis Chapter | Mathematical Content | Corresponding Code |
|---|---|---|
| Ch. 2 — Flow Theory | ODE: dx_t/dt = v_t, OT-CFM objective | `f5_tts/model/` (DiT forward pass) |
| Ch. 2 — ODE Discretization | Euler truncation error O(h^2) | `odeint(..., method="euler")` in `_teacher_sampling` |
| Ch. 3 — Duration Drift | P(Y,T|X) = P(T|X)*P(Y|X,T), velocity dilation | `predict_duration()` then alpha computation |
| Ch. 3 — Boundary Collision | C0 discontinuity at `teacher_stopping_time` | `_teacher_sampling()` hard return statement |
| Ch. 4 — TAGS-Flow | w(t) cosine manifold, hybrid v_hybrid | `_tags_flow_sampling()` [NEW METHOD] |
| Ch. 4 — GRPO Theory | GRPO loss, 1.5k reward-hacking boundary | `grpo_duration_trainer.py` (existing, reference only) |
| Ch. 5 — Experiments | WER/SIM/RTF/UTMOS tables | `demo.ipynb` + evaluation scripts |

---

## PART 12: Key Hyperparameters

| Parameter | DMOSpeech 2 Default | TAGS-Flow Range | Effect |
|---|---|---|---|
| `teacher_stopping_time` | `0.07` | replaced by `t1`, `t2` | transition window boundaries |
| `t1` | N/A (new) | `0.15` to `0.25` | start of blend zone |
| `t2` | N/A (new) | `0.30` to `0.45` | end of blend zone |
| `delta` (alpha clamp) | N/A (new) | `0.10` to `0.20` | max velocity rescaling allowed |
| `N` (total steps) | teacher:16, student:4 | `4` to `8` unified | speed vs. quality tradeoff |
| `eta` | `1.0` | `0.0` to `1.0` | DDIM (0) vs. DDPM (1) stochasticity |
| `cfg_strength` | `2.0` | `1.5` to `3.0` | classifier-free guidance strength |
| `sway_coefficient` | `-1.0` | `-1.0` to `0.0` | time-step non-uniformity |

---

*End of Context Brief. The model reading this should now be able to understand, modify, and extend `src/infer.py` without needing to explore the codebase independently.*
