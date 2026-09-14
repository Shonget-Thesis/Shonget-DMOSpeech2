# 🧮 Baseline DMOSpeech 2: Manual Numerical Calculation & Casio fx-570CW Keystroke Guide

**A Step-by-Step Defense & Hand-Calculation Guide for Thesis Examiners and Baseline Verification**  
**Target Calculator:** Casio **fx-570CW** / **fx-991CW** (ClassWiz CW Series)  
**System:** Original Baseline DMOSpeech 2 (Guo et al., July 2025)  
**Input Benchmark Audio:** `src/f5_tts/infer/examples/basic/basic_ref_en.wav` (*"Some call me nature, others call me mother nature."*)  
**Input Benchmark Text:** `"Hello! DMOSpeech 2 is successfully running on this system."`  
**Verified Source Log:** [`eval_results/baseline_computational_step_log.json`](eval_results/baseline_computational_step_log.json)  
**Generated Audio:** [`eval_results/01_baseline_dmospeech2.wav`](eval_results/01_baseline_dmospeech2.wav) / [`output.wav`](output.wav)

---

## 🧭 Casio fx-570CW Quick Setup & Key Legend

Before executing baseline calculations, configure your calculator into standard decimal mode:

1. Press **`[HOME]`**
2. Use arrow keys `[◄]` `[►]` `[▲]` `[▼]` to highlight **`Calculate`**
3. Press **`[OK]`** or **`[EXE]`**
4. Set Angle Unit to **Radian** (Required for the Sway Schedule and Hann Window in Steps 2 & 6):
   * Press **`[SETTINGS]`** (Gear icon)
   * Select **`Calc Settings`** $\to$ press **`[OK]`**
   * Select **`Angle Unit`** $\to$ press **`[OK]`**
   * Select **`Radian`** $\to$ press **`[OK]`**
   * Press **`[AC]`** or **`[BACK]`** to return to the active calculation line.

> 💡 **Casio fx-570CW Decimal Display Tip:**  
> If an answer displays as a fraction or radical, press **`[FORMAT]`**, select **`Decimal`**, and press **`[EXE]`**.

---

## 📌 STEP 1: Audio Loading & Root-Mean-Square (RMS) Energy

### 1.1 Formula:
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$

### 1.2 Real Values Plugged In:
* Total samples $N = 124,416$ samples at $f_s = 24,000\text{ Hz}$ ($5.184\text{ seconds}$)
* Sum of squared audio samples: $\sum_{i=1}^{124,416} x_i^2 = \mathbf{2,125.7955}$

---

### 1.3 Step A: Mean Square Energy Calculation
$$\text{Mean Square} = \frac{2125.7955}{124416}$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[2] [1] [2] [5] [.] [7] [9] [5] [5] [÷] [1] [2] [4] [4] [1] [6] [EXE]
```
*(Or use fraction key: `[x/y]` `2125.7955` `[▼]` `124416` `[EXE]`)*

* **Screen Output:** `0.01708619...`

---

### 1.4 Step B: Square Root to Obtain RMS
$$\text{RMS} = \sqrt{\text{Ans}} = \sqrt{0.01708619...} = \mathbf{0.13071}$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[√] [Ans] [EXE]
```
*(Or single-line formula:)*
```
[√] [(] [2] [1] [2] [5] [.] [7] [9] [5] [5] [÷] [1] [2] [4] [4] [1] [6] [)] [EXE]
```

* **Screen Output:** `0.130714159...` $\approx \mathbf{0.13071}$

#### 🔍 Threshold Rule Check:
$$\text{Rule: If } \text{RMS} < 0.10, \quad \text{scale} = \frac{0.10}{\text{RMS}}, \quad \text{else } \text{scale} = 1.0$$
* Is $0.13071 < 0.10000$? **No.**
* Scaling Multiplier = $\mathbf{1.0}$ (Audio is passed directly at unity gain).

---

## 📌 STEP 2: STFT & Mel Spectrogram Matrix Transformation

### 2.1 Reflection Padding
* Formula:
  $$\text{padding} = \frac{N_{\text{fft}} - \text{hop}}{2} = \frac{1024 - 256}{2}$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[(] [1] [0] [2] [4] [-] [2] [5] [6] [)] [÷] [2] [EXE]
```
* **Screen Output:** $\mathbf{384}$ samples per edge.

Total symmetrical padded length:
$$N_{\text{padded}} = 124416 + 384 + 384$$
```
[1] [2] [4] [4] [1] [6] [+] [3] [8] [4] [+] [3] [8] [4] [EXE]
```
* **Screen Output:** $\mathbf{125184}$ samples.

---

### 2.2 Time Framing & Frequency Bins
* Frame count $M$:
  $$M = \frac{N_{\text{padded}} - N_{\text{fft}}}{\text{hop}} + 1 = \frac{125184 - 1024}{256} + 1$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[(] [1] [2] [5] [1] [8] [4] [-] [1] [0] [2] [4] [)] [÷] [2] [5] [6] [+] [1] [EXE]
```
* **Screen Output:** $\mathbf{486}$ mel frames.

* Linear FFT Bins $K$:
  $$K = \frac{N_{\text{fft}}}{2} + 1 = \frac{1024}{2} + 1$$
```
[1] [0] [2] [4] [÷] [2] [+] [1] [EXE]
```
* **Screen Output:** $\mathbf{513}$ frequency bins ($0\text{ Hz}$ to $12,000\text{ Hz}$).

---

### 2.3 Periodic Hann Window at Center ($n = 512$)
$$w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{1024}\right) \quad \text{at } n = 512$$

#### 🔘 Casio fx-570CW Keystroke Sequence (In Radian Mode):
```
[0] [.] [5] [-] [0] [.] [5] [cos] [(] [2] [SHIFT] [7] [×] [5] [1] [2] [÷] [1] [0] [2] [4] [)] [EXE]
```
*(Note: On fx-570CW, $\pi$ is entered via `[SHIFT]` `[7]` or `[SHIFT]` `[×10^x]`)*

* **Screen Output:** $\mathbf{1.0000}$ (Peak transmission at frame center).

---

## 📌 STEP 3: GRPO Duration Predictor Policy Network ($\pi_\phi$)

### 3.1 Neural Duration Policy
* Combined prompt + target text length: $112$ tokens.
* Duration policy $\pi_\phi$ selects class index: $52$.
* Total predicted frames:
  $$L = 52 \times 10 = \mathbf{520\text{ mel frames}}$$

---

### 3.2 Conversion to Physical Speech Duration
$$\text{Duration (seconds)} = \frac{L \times \text{hop}}{f_s} = \frac{520 \times 256}{24000}$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[5] [2] [0] [×] [2] [5] [6] [÷] [2] [4] [0] [0] [0] [EXE]
```
*(If displayed as fraction, press `[FORMAT]` $\to$ `Decimal` $\to$ `[EXE]`)*

* **Screen Output:** `5.546666667` $\approx \mathbf{5.5467\text{ seconds}}$

---

## 📌 STEP 4: Mel Rescaling & Zero Sequence Alignment

### 4.1 Checkpoint Scale Factor Normalization
DMOSpeech 2 divides all log-mel energies by the learned checkpoint scale $\gamma = \mathbf{2.9019215}$:

* **Maximum Mel Value ($+1.542047$):**
```
[1] [.] [5] [4] [2] [0] [4] [7] [÷] [2] [.] [9] [0] [1] [9] [2] [1] [5] [EXE]
```
* **Screen Output:** `0.531388...` $\approx \mathbf{+0.53139}$

* **Minimum Mel Value ($-3.967345$):**
```
[(-)] [3] [.] [9] [6] [7] [3] [4] [5] [÷] [2] [.] [9] [0] [1] [9] [2] [1] [5] [EXE]
```
* **Screen Output:** `-1.367144...` $\approx \mathbf{-1.36714}$

---

### 4.2 Sequence Difference for Padding
$$\Delta L = 520 - 486$$
```
[5] [2] [0] [-] [4] [8] [6] [EXE]
```
* **Screen Output:** $\mathbf{34\text{ frames}}$ of zeros appended to create conditioning tensor of shape `[1, 520, 100]`.

---

## 📌 STEP 5: Gaussian Latent Prior ($y_0$) at $t = 0$

* Shape: $1 \times 520 \times 100 = \mathbf{52,000\text{ elements}}$
* Sample Mean: $\mathbf{+0.001602}$, Standard Deviation: $\mathbf{0.999511}$
* L2 Vector Norm across all elements:
  $$\|y_0\|_2 = \sqrt{\sum_{i=1}^{52,000} y_i^2} = \mathbf{228.1634}$$

---

## 📌 STEP 6: Sway Schedule Time Grid Calculation (Cosine Sway $s = -1.0$)

DMOSpeech 2 computes its integration schedule using the nonlinear Cosine Sway formula:
$$t_{\text{sway}} = t_{\text{raw}} - \left[\cos\left(\frac{\pi}{2} t_{\text{raw}}\right) - 1 + t_{\text{raw}}\right]$$

For an $N=8$ teacher schedule, the uniform time intervals are $t_{\text{raw}} = \left[0.0, \frac{1}{8}, \frac{2}{8}, \frac{3}{8}\right] = [0.000, 0.125, 0.250, 0.375]$.

---

### 6.1 Step 1 Time Point ($t_{\text{raw}} = 0.125$)
$$t_{\text{sway}, 1} = 0.125 - \left[\cos\left(\frac{\pi}{2} \times 0.125\right) - 1 + 0.125\right]$$

#### 🔘 Casio fx-570CW Keystroke Sequence (Ensure Radian Mode!):
```
[0] [.] [1] [2] [5] [-] [(] [cos] [(] [SHIFT] [7] [÷] [2] [×] [0] [.] [1] [2] [5] [)] [-] [1] [+] [0] [.] [1] [2] [5] [)] [EXE]
```
* **Screen Output:** `0.0192147...` $\approx \mathbf{0.019215}$

---

### 6.2 Step 2 Time Point ($t_{\text{raw}} = 0.250$)
$$t_{\text{sway}, 2} = 0.250 - \left[\cos\left(\frac{\pi}{2} \times 0.250\right) - 1 + 0.250\right]$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[0] [.] [2] [5] [0] [-] [(] [cos] [(] [SHIFT] [7] [÷] [2] [×] [0] [.] [2] [5] [0] [)] [-] [1] [+] [0] [.] [2] [5] [0] [)] [EXE]
```
* **Screen Output:** `0.0761204...` $\approx \mathbf{0.076120}$

---

### 6.3 Teacher Cutoff Rule Test
$$\text{Condition: Teacher stops when } t > 0.07000$$
* Is $t_2 = 0.076120 > 0.07000$? **Yes.**
* The baseline halts teacher DiT integration at:
  $$t_{\text{final}} = \mathbf{0.076120}$$

---

## 📌 STEP 7: Teacher Euler Step Sizes

### 7.1 Teacher Step 1 Interval ($\Delta t_1$):
$$\Delta t_1 = 0.019215 - 0.00000 = \mathbf{0.019215}$$

### 7.2 Teacher Step 2 Interval ($\Delta t_2$):
$$\Delta t_2 = 0.076120 - 0.019215$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[0] [.] [0] [7] [6] [1] [2] [0] [-] [0] [.] [0] [1] [9] [2] [1] [5] [EXE]
```
* **Screen Output:** $\mathbf{0.056905}$

---

## 📌 STEP 8: The Unstated Handover Cliff & Jump Extrapolation (DMOSpeech 2 Flaw)

This is the exact mathematical flaw exposed in your thesis:

```
Baseline DMOSpeech 2 Handover at t = 0.076120:
Teacher stops abruptly ───► Extrapolates via linear tangent:
                            Δt_remaining = 1.0 - 0.076120 = 0.923880
                            test_out = x_2 + 0.923880 · v_boundary
                                      │
                                      ▼  💥 UNREGULARIZED COLLISION!
                            x_1 = output_student (w_cushion = 1.00)
                            • 0% shock absorbed (0.00 units)
                            • Full 359.68 displacement crashes into latent space
                            • Causes 19.88 Spectral Flux spike!
```

---

### 8.1 Remaining Time Horizon Extrapolation
$$\Delta t_{\text{remaining}} = 1.0 - t_{\text{final}} = 1.0 - 0.076120$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[1] [-] [0] [.] [0] [7] [6] [1] [2] [0] [EXE]
```
* **Screen Output:** $\mathbf{0.923880}$

---

### 8.2 Jump Delta Magnitude Calculation
Teacher velocity boundary norm $\|\mathbf{v}_{\text{boundary}}\|_2 = \mathbf{240.23}$.
$$\|\mathbf{jump\_delta}\|_2 = 0.923880 \times 240.23$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[0] [.] [9] [2] [3] [8] [8] [0] [×] [2] [4] [0] [.] [2] [3] [EXE]
```
* **Screen Output:** `221.9436...` $\approx \mathbf{221.9478}$

---

### 8.3 Student Timestep Swaying Grid ($s_{\text{raw}} = 0.25, 0.50, 0.75$)

#### Student Step 1: $s_1 = \mathbf{0.076120}$
* Interpolation Weights:
  * Noise weight: $1 - 0.076120 = \mathbf{0.923880}$ ($92.39\%$)
  * State weight: $\mathbf{0.076120}$ ($7.61\%$)

#### Student Step 2 ($s_{\text{raw}} = 0.50$):
$$s_2 = 0.50 - \left[\cos\left(\frac{\pi}{2} \times 0.50\right) - 1 + 0.50\right]$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[0] [.] [5] [-] [(] [cos] [(] [SHIFT] [7] [÷] [2] [×] [0] [.] [5] [)] [-] [1] [+] [0] [.] [5] [)] [EXE]
```
* **Screen Output:** `0.292893...` $\approx \mathbf{0.292893}$
* Noise weight: $1 - 0.292893 = \mathbf{0.707107}$ ($70.71\%$)

#### Student Step 3 ($s_{\text{raw}} = 0.75$):
$$s_3 = 0.75 - \left[\cos\left(\frac{\pi}{2} \times 0.75\right) - 1 + 0.75\right]$$

#### 🔘 Casio fx-570CW Keystroke Sequence:
```
[0] [.] [7] [5] [-] [(] [cos] [(] [SHIFT] [7] [÷] [2] [×] [0] [.] [7] [5] [)] [-] [1] [+] [0] [.] [7] [5] [)] [EXE]
```
* **Screen Output:** `0.617316...` $\approx \mathbf{0.617317}$
* Noise weight: $1 - 0.617317 = \mathbf{0.382683}$ ($38.27\%$)

---

### 8.4 The Baseline Shock Accounting
In baseline DMOSpeech 2, the displacement vector is **not regularized** ($w_{\text{cushion}} = 1.00$):
$$\text{Absorbed Shock} = (1.0 - 1.00) \times \|\Delta x\|_2 = 0.00 \times 359.6839 = \mathbf{0.00\text{ units}}$$
* **Result:** **0% shock absorption**. The full $359.68$ vector displacement collides directly into the student's next step, exciting synthetic boundary phase jitter.

---

## 📌 STEP 9: Vocos Waveform Decoding & Baseline Acoustic Turbulence

### 9.1 Overlap-Add Audio Sample Calculation
* Generated target frames: $520$
* Hop length: $256$
* Total synthesized samples before trimming:
  $$N_{\text{raw}} = 520 \times 256 = \mathbf{133,120\text{ samples}}$$

* Trimming filterbank delay ($520$ samples):
  $$N_{\text{final}} = 133,120 - 520$$
```
[1] [3] [3] [1] [2] [0] [-] [5] [2] [0] [EXE]
```
* **Screen Output:** $\mathbf{132600}$ PCM audio samples.

* Final baseline audio duration:
  $$\text{Duration} = \frac{132600}{24000}$$
```
[1] [3] [2] [6] [0] [0] [÷] [2] [4] [0] [0] [0] [EXE]
```
* **Screen Output:** $\mathbf{5.525\text{ seconds}}$  
*(Saved directly to [`eval_results/01_baseline_dmospeech2.wav`](eval_results/01_baseline_dmospeech2.wav) and [`output.wav`](output.wav)).*

---

### 9.2 Baseline Empirical Turbulence Metrics
Because the baseline does not absorb boundary shock:
* **Baseline Transition Spectral Flux:** **`19.8819`** (High turbulence boundary)
* **Baseline Maximum Boundary Spike:** **`87.9715`** (Instantaneous velocity shock)

*(Compared to our proposed system's `13.0673` flux and `61.2432` spike, proving the 34.3% turbulence reduction).*

---

## 📋 Defense Cheat Sheet: Quick Button Summary for Panelists (Baseline)

| Calculation / Concept | Keystroke Input on Casio fx-570CW | Screen Output |
| :--- | :--- | :---: |
| **1. Audio RMS** | `[√]` `[(]` `2125.7955` `[÷]` `124416` `[)]` `[EXE]` | **`0.13071`** |
| **2. Padded STFT Length** | `124416` `[+]` `384` `[+]` `384` `[EXE]` | **`125184`** |
| **3. STFT Frames** | `[(]` `125184` `[-]` `1024` `[)]` `[÷]` `256` `[+]` `1` `[EXE]` | **`486`** |
| **4. Speech Duration** | `520` `[×]` `256` `[÷]` `24000` `[EXE]` | **`5.5467 s`** |
| **5. Sway Step 2 ($s_1$)** | `0.25` `[-]` `[(]` `cos(π/2 × 0.25)` `[-]` `1` `[+]` `0.25` `[)]` `[EXE]` | **`0.07612`** |
| **6. Remaining Horizon** | `1.0` `[-]` `0.076120` `[EXE]` | **`0.92388`** |
| **7. Jump Delta** | `0.923880` `[×]` `240.23` `[EXE]` | **`221.9478`** |
| **8. Absorbed Shock** | `(1.0 - 1.00)` `[×]` `359.6839` `[EXE]` | **`0.00 units (Crash!)`** |
| **9. Final Wave Samples** | `(520 × 256)` `[-]` `520` `[EXE]` | **`132600 samples`** |
| **10. Final Audio Duration**| `132600` `[÷]` `24000` `[EXE]` | **`5.525 s`** |
