# 🧮 DMOSpeech 2: Fully Computed Numerical Calculation Walkthrough

**A Concrete, Real-Input Step-by-Step Mathematical Derivation Showing the Exact Numbers Plugged into Every Single Formula.**

* **Prompt Audio:** `src/f5_tts/infer/examples/basic/basic_ref_en.wav` (*"Some call me nature, others call me mother nature."*)  
* **Target Text:** `"Hello! DMOSpeech 2 is successfully running on this system."`  
* **Source Log:** [`eval_results/baseline_computational_step_log.json`](eval_results/baseline_computational_step_log.json)  
* **Generated Baseline Audio:** [`eval_results/01_baseline_dmospeech2.wav`](eval_results/01_baseline_dmospeech2.wav) / [`output.wav`](output.wav)  

---

## 📌 STEP 1: Audio Loading & Exact RMS Calculation

### Formula:
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$

### Real Values Plugged In:
* The input file has $N = 124,416$ samples at $f_s = 24,000\text{ Hz}$ (duration = $5.184\text{ seconds}$).
* Sum of squared audio samples:
  $$\sum_{i=1}^{124,416} x_i^2 = \mathbf{2,125.7955}$$
* Mean of squares:
  $$\frac{2,125.7955}{124,416} = \mathbf{0.017086}$$
* Square root:
  $$\text{RMS} = \sqrt{0.017086} = \mathbf{0.13071}$$

### Threshold Check:
$$\text{Threshold Rule: If } \text{RMS} < 0.10, \quad \text{scale} = \frac{0.10}{\text{RMS}}$$
* Plugged In: Is $0.13071 < 0.10000$? **No.**
* Scaling Multiplier:
  $$\mathbf{x}_{\text{norm}} = \mathbf{x} \times \mathbf{1.0} \implies \text{Normalized RMS} = \mathbf{0.13071}$$

---

## 📌 STEP 2: STFT & Mel Spectrogram Matrix Transformation

### 2.1 Reflection Padding
* Formula:
  $$\text{padding} = \frac{N_{\text{fft}} - \text{hop}}{2} = \frac{1024 - 256}{2} = \mathbf{384\text{ samples}}$$
* Symmetrical padding on both ends:
  $$N_{\text{padded}} = 124,416 + 384 + 384 = \mathbf{125,184\text{ samples}}$$

### 2.2 Time Framing & Frequency Binning
* Frame count calculation:
  $$M = \frac{N_{\text{padded}} - N_{\text{fft}}}{\text{hop}} + 1 = \frac{125,184 - 1024}{256} + 1 = \mathbf{486\text{ frames}}$$
* Linear FFT frequency bins:
  $$K = \frac{N_{\text{fft}}}{2} + 1 = \frac{1024}{2} + 1 = \mathbf{513\text{ bins}} \quad (0\text{ Hz to } 12,000\text{ Hz})$$

### 2.3 Periodic Hann Window
* Formula:
  $$w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{1024}\right), \quad n \in [0, 1023]$$
  * At center $n = 512$: $w(512) = 0.5 - 0.5\cos(\pi) = 0.5 - (-0.5) = \mathbf{1.000}$
  * At edges $n = 0$: $w(0) = 0.5 - 0.5(1) = \mathbf{0.000}$

### 2.4 Linear Magnitude Spectrogram
* Formula:
  $$|X(k, m)| = \sqrt{\operatorname{Re}(X)^2 + \operatorname{Im}(X)^2 + 10^{-9}}$$
* Resulting matrix shape: **`[513, 486]`** (513 frequency rows $\times$ 486 time columns).

### 2.5 Triangular Mel Matrix Multiply & Log Compression
* The precomputed triangular filterbank matrix has shape **`[100, 513]`** (mapping 513 linear frequencies to 100 human-perceptual Mel bands).
* Matrix multiplication:
  $$S_{\text{mel}}[100, 486] = \mathbf{W}_{\text{mel}}[100, 513] \times |X|[513, 486]$$
* Log Compression Formula:
  $$S_{\text{log\_mel}} = \ln(\max(S_{\text{mel}}, 10^{-5}))$$
* **Actual Real Values in Our Resulting Tensor:**
  * Shape: `[1, 486, 100]`
  * Minimum: `-3.967345` | Maximum: `+1.542047`
  * Mean: `-0.395779` | Standard Deviation: `0.777149`
  * L2 Vector Norm:
    $$\|S_{\text{log\_mel}}\|_2 = \sqrt{\sum_{i=1}^{48,600} S_i^2} = \mathbf{198.8735}$$

---

## 📌 STEP 3: Duration Predictor Policy Network ($\pi_\phi$)

### Text Token Count:
* Combined Text: `"Some call me nature, others call me mother nature. Hello! DMOSpeech 2 is successfully running on this system."`
* Mapped into token indices: $112$ tokens.

### Neural Policy Duration Formula:
$$L = \operatorname{argmax}\left(\operatorname{Softmax}\left(\pi_\phi(\text{mel}_{\text{prompt}}, \text{text})\right)\right) \times 10$$
* The policy network selects class index $52$:
  $$L = 52 \times 10 = \mathbf{520\text{ mel frames}}$$

### Converting Frames to Physical Audio Seconds:
$$\text{Duration (seconds)} = \frac{L \times \text{hop}}{\text{sample\_rate}} = \frac{520 \times 256}{24,000} = \frac{133,120}{24,000} = \mathbf{5.54667\text{ seconds}}$$

---

## 📌 STEP 4: Mel Rescaling & Zero Padding

### Scale Normalization Formula:
$$\mathbf{C} = \frac{S_{\text{log\_mel}}}{\gamma} \quad \text{where Learned Checkpoint Scale } \gamma = \mathbf{2.9019215}$$
* Every mel value is divided by $2.9019215$:
  * Max value: $\frac{1.542047}{2.9019215} = \mathbf{+0.53139}$
  * Min value: $\frac{-3.967345}{2.9019215} = \mathbf{-1.36714}$

### Sequence Alignment & Padding:
* Prompt frames: $486$ | Target frames: $520$
* Difference:
  $$\Delta L = 520 - 486 = \mathbf{34\text{ generated frames}}$$
* Padding: $34$ frames of zeros are appended to the end of the conditioning tensor.
* Conditioning Tensor `step_cond`: Shape `[1, 520, 100]`, L2 Norm = **`198.8735`**.

---

## 📌 STEP 5: Gaussian Prior Noise Sampling ($y_0$)

At time $t = 0$, pure white noise is sampled:
$$y_0 \sim \mathcal{N}(0, 1) \in \mathbb{R}^{1 \times 520 \times 100}$$
* Total individual float values: $1 \times 520 \times 100 = \mathbf{52,000\text{ values}}$.
* **Actual Real Values Sampled in This Run:**
  * Sample Mean: $\mathbf{+0.001602}$
  * Sample Standard Deviation: $\mathbf{0.999511}$
  * L2 Vector Norm:
    $$\|y_0\|_2 = \sqrt{\sum_{i=1}^{52,000} y_i^2} = \mathbf{228.1634}$$

---

## 📌 STEP 6: Sway Schedule Time Grid Calculation

### General Cosine Sway Formula ($s = -1.0$):
$$t_{\text{sway}} = t_{\text{raw}} - \left[\cos\left(\frac{\pi}{2} t_{\text{raw}}\right) - 1 + t_{\text{raw}}\right]$$

### Manual Step-by-Step Arithmetic for Every Grid Point ($N = 8$ Steps):

1. **For Step $0$ ($t_{\text{raw}} = 0.000$):**
   * $\cos(0) = 1.0$
   * $t_{\text{sway}, 0} = 0 - [1.0 - 1 + 0] = \mathbf{0.00000}$

2. **For Step $1$ ($t_{\text{raw}} = 0.125$):**
   * Angle: $\frac{\pi}{2} \times 0.125 = 0.19635\text{ rad}$
   * $\cos(0.19635) = 0.980785$
   * Bracket: $[0.980785 - 1.0 + 0.125] = 0.105785$
   * $t_{\text{sway}, 1} = 0.125 - 0.105785 = \mathbf{0.019215}$

3. **For Step $2$ ($t_{\text{raw}} = 0.250$):**
   * Angle: $\frac{\pi}{2} \times 0.250 = 0.39270\text{ rad}$
   * $\cos(0.39270) = 0.923880$
   * Bracket: $[0.923880 - 1.0 + 0.250] = 0.173880$
   * $t_{\text{sway}, 2} = 0.250 - 0.173880 = \mathbf{0.076120}$

4. **For Step $3$ ($t_{\text{raw}} = 0.375$):**
   * Angle: $\frac{\pi}{2} \times 0.375 = 0.58905\text{ rad}$
   * $\cos(0.58905) = 0.831470$
   * Bracket: $[0.831470 - 1.0 + 0.375] = 0.206470$
   * $t_{\text{sway}, 3} = 0.375 - 0.206470 = \mathbf{0.168530}$

### Cutoff Test:
* Teacher stops when $t > 0.07000$.
* Is $t_2 = 0.076120 > 0.07$? **Yes.**
* Active Teacher Grid:
  $$t_{\text{teacher}} = [0.00000, 0.019215, 0.076120]$$

---

## 📌 STEP 7: Teacher DiT Euler ODE Integration Steps

### Classifier-Free Guidance (CFG) Formula:
$$\mathbf{v}_{\text{cfg}} = \mathbf{v}_{\text{cond}} + 2.0 \cdot (\mathbf{v}_{\text{cond}} - \mathbf{v}_{\text{uncond}})$$

### 7.1 Euler Step 1 ($t = 0.00000 \to 0.019215$):
* Step size:
  $$\Delta t_1 = 0.019215 - 0.00000 = \mathbf{0.019215}$$
* Current State Norm: $\|x_0\|_2 = \mathbf{228.16}$
* Model forward pass calculates velocities:
  * Raw predicted velocity norm: $\|\mathbf{v}_{\text{cond}}\|_2 = \mathbf{239.38}$
  * CFG guidance velocity norm: $\|\mathbf{v}_{\text{cfg}}\|_2 = \mathbf{244.60}$
* Euler State Update:
  $$x_1 = x_0 + \Delta t_1 \cdot \mathbf{v}_{\text{cfg}} = x_0 + 0.019215 \cdot \mathbf{v}_{\text{cfg}}$$
* Resulting State Norm:
  $$\|x_1\|_2 = \mathbf{227.60}$$

### 7.2 Euler Step 2 ($t = 0.019215 \to 0.076120$):
* Step size:
  $$\Delta t_2 = 0.076120 - 0.019215 = \mathbf{0.056905}$$
* Current State Norm: $\|x_1\|_2 = \mathbf{227.60}$
* CFG guidance velocity norm: $\|\mathbf{v}_{\text{cfg}}\|_2 = \mathbf{241.67}$
* Euler State Update:
  $$x_2 = x_1 + \Delta t_2 \cdot \mathbf{v}_{\text{cfg}} = x_1 + 0.056905 \cdot \mathbf{v}_{\text{cfg}}$$
* Resulting State Norm:
  $$\|x_2\|_2 = \mathbf{225.40}$$

---

## 📌 STEP 8: The Unstated Cutoff & Stochastic Student Refinement

### 8.1 The Linear Jump Extrapolation (DMOSpeech 2's Flaw)
* Formula in `src/infer.py`:
  $$\Delta t_{\text{remaining}} = 1.0 - t_{\text{final}} = 1.0 - 0.076120 = \mathbf{0.923880}$$
  $$\mathbf{test\_out} = x_2 + \Delta t_{\text{remaining}} \cdot \mathbf{v}_{\text{boundary}}$$
* Real Values Plugged In:
  * Teacher final velocity norm: $\|\mathbf{v}_{\text{boundary}}\|_2 = \mathbf{240.23}$
  * Jump Vector Delta:
    $$\|\mathbf{jump\_delta}\|_2 = 0.923880 \times 240.23 = \mathbf{221.9478}$$
  * Extrapolated State:
    $$\|\mathbf{test\_out}\|_2 = \mathbf{302.3214}$$

---

### 8.2 Student Discrete Steps with Stochastic Noise Shock

Student timesteps are swayed with $s_{\text{raw}} = [0.25, 0.50, 0.75]$:
$$s_1 = \mathbf{0.076120}, \quad s_2 = \mathbf{0.292893}, \quad s_3 = \mathbf{0.617317}$$

#### Student Pass 1 ($s_1 = 0.076120$):
1. Samples fresh Gaussian noise $x_0 \sim \mathcal{N}(0, 1)$, with $\|x_0\|_2 = \mathbf{227.94}$.
2. Linear interpolation weights:
   $$\text{Noise Weight } (1 - s_1) = 1.0 - 0.076120 = \mathbf{0.923880} \quad (\mathbf{92.39\%})$$
   $$\text{State Weight } s_1 = \mathbf{0.076120} \quad (\mathbf{7.61\%})$$
3. Interpolated latent state:
   $$\phi_1 = 0.923880 \cdot x_0 + 0.076120 \cdot \mathbf{test\_out} \implies \|\phi_1\|_2 = \mathbf{212.86}$$
4. Student generator residual prediction: $\|\text{pred}_1\|_2 = \mathbf{98.15}$
5. Refined Mel Output:
   $$\text{output}_1 = \phi_1 + (1 - s_1) \cdot \text{pred}_1 = \phi_1 + 0.923880 \cdot \text{pred}_1$$
   * Resulting Norm: $\|\text{output}_1\|_2 = \mathbf{302.61}$$

#### Student Pass 2 ($s_2 = 0.292893$):
* Weights: Noise Weight $= 1 - 0.292893 = \mathbf{0.707107}$ ($70.71\%$), State Weight $= \mathbf{0.292893}$ ($29.29\%$).
* Refined Mel Output Norm: $\|\text{output}_2\|_2 = \mathbf{309.08}$$

#### Student Pass 3 ($s_3 = 0.617317$):
* Weights: Noise Weight $= 1 - 0.617317 = \mathbf{0.382683}$ ($38.27\%$), State Weight $= \mathbf{0.617317}$ ($61.73\%$).
* Final Predicted Mel Output Norm: $\|\text{output}_3\|_2 = \mathbf{311.21}$$

---

## 📌 STEP 9: Vocos Neural Vocoder Waveform Synthesis

### 9.1 Mel Rescaling Formula:
$$\text{Mel}_{\text{final}} = \text{output}_3^{\top} \times \gamma = \text{output}_3^{\top} \times \mathbf{2.9019215}$$
* Shape: `[1, 100, 1006]` (486 prompt frames + 520 target generation frames)

### 9.2 Truncating Prompt Conditioning
* Slices off the 486 prompt conditioning frames to isolate the target speech:
  $$\text{Mel}_{\text{target}} = \text{Mel}_{\text{final}}[:, :, 486:] \implies \text{Shape: } [1, 100, 520]$$

### 9.3 Overlap-Add Waveform Synthesis
* 8 depthwise ConvNeXt blocks convert 520 mel frames into complex Fourier bins $Z \in \mathbb{C}^{513 \times 520}$.
* Inverse STFT synthesis:
  $$N_{\text{wave}} = 520 \text{ frames} \times 256\text{ hop} = 133,120\text{ samples}$$
* Edge padding trimming:
  $$N_{\text{final}} = 133,120 - 520 = \mathbf{132,600\text{ samples}}$$
* Final Audio Duration:
  $$\text{Duration} = \frac{132,600}{24,000} = \mathbf{5.525\text{ seconds}}$$
* Full Audio Saved to: [`eval_results/01_baseline_dmospeech2.wav`](eval_results/01_baseline_dmospeech2.wav) / [`output.wav`](output.wav) (File size: 265,260 bytes).
