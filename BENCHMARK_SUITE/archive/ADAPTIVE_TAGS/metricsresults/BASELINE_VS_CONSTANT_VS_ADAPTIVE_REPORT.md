# 🔬 Complete Tri-Way Benchmark: Baseline vs. Constant Cushion (0.85) vs. Adaptive TAGS-Flow

**Direct Acoustic Turbulence, Dynamic Cushioning, and Intelligibility Audit Across 5 Diverse Voices**

| # | Speaker Profile | Baseline Flux ($w=1.0$) | Constant Flux ($w=0.85$) | Adaptive Flux ($w_{\text{adapt}}$) | Dynamic $w$ | Constant Red. | Adaptive Red. | Audio Links |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | **Female Nature**<br>*Soft, calm, serene feminine British/American English* | `24.72` | **`30.1525`** | **`20.22`** | **`0.8498`** | ≈ **`-22.0%`** | 🏆 **`18.2%`** | [🔴 Baseline](../output_baseline/sample_1_female_nature.wav)<br>[🟡 Constant](../output_constant/sample_1_female_nature.wav)<br>[🟢 Adaptive](../output/sample_1_female_nature.wav) |
| **2** | **Male Country**<br>*Rustic, folksy, gritty Southern male storytelling dialect* | `16.45` | **`0.5527`** | **`15.81`** | **`0.841`** | 🏆 **`96.6%`** | 🏆 **`3.9%`** | [🔴 Baseline](../output_baseline/sample_2_male_country.wav)<br>[🟡 Constant](../output_constant/sample_2_male_country.wav)<br>[🟢 Adaptive](../output/sample_2_male_country.wav) |
| **3** | **Male Broadcast Narrator**<br>*Authoritative, crisp, professional documentary narration* | `13.61` | **`38.189`** | **`18.63`** | **`0.8469`** | ≈ **`-180.6%`** | ≈ **`-36.9%`** | [🔴 Baseline](../output_baseline/sample_3_male_broadcast_narrator.wav)<br>[🟡 Constant](../output_constant/sample_3_male_broadcast_narrator.wav)<br>[🟢 Adaptive](../output/sample_3_male_broadcast_narrator.wav) |
| **4** | **Male Academic Lecture**<br>*Fast-paced, intellectual, analytical scientific lecture* | `9.5` | **`30.9478`** | **`14.33`** | **`0.8291`** | ≈ **`-225.8%`** | ≈ **`-50.8%`** | [🔴 Baseline](../output_baseline/sample_4_male_academic_lecture.wav)<br>[🟡 Constant](../output_constant/sample_4_male_academic_lecture.wav)<br>[🟢 Adaptive](../output/sample_4_male_academic_lecture.wav) |
| **5** | **Deep Dramatic Cinematic Voice**<br>*Dark, gravelly, intense theatrical male voice* | `18.6` | **`0.007`** | **`1.62`** | **`0.8793`** | 🏆 **`100.0%`** | 🏆 **`91.3%`** | [🔴 Baseline](../output_baseline/sample_5_deep_dramatic_cinematic_voice.wav)<br>[🟡 Constant](../output_constant/sample_5_deep_dramatic_cinematic_voice.wav)<br>[🟢 Adaptive](../output/sample_5_deep_dramatic_cinematic_voice.wav) |


---

## 📋 Intelligibility Comparison (OpenAI Whisper ASR):

### Sample 1: Female Nature (Soft, calm, serene feminine British/American English)
* **Target Text:** "Hello! This is a test of the adaptive smooth speech synthesis system."
* **🔴 Baseline Transcript:** "Hello! This is a test of the adaptive smooth speech synthesis system."
* **🟡 Constant (0.85) Transcript:** "Hello, this is a test of the adaptive smooth speech synthesis system."
* **🟢 Adaptive Transcript:** "Hello, this is a test of the adaptive smooth speech synthesis system."

### Sample 2: Male Country (Rustic, folksy, gritty Southern male storytelling dialect)
* **Target Text:** "Well howdy there! The quick brown fox jumps right over that lazy dog."
* **🔴 Baseline Transcript:** "Well howdy there! The quick brown fox jumps right over that lazy dog."
* **🟡 Constant (0.85) Transcript:** "Well howdy there! The quick brown fox jumps right over that lazy dog."
* **🟢 Adaptive Transcript:** "Well, howdy there! The quick brown fox jumps right over that lazy dog."

### Sample 3: Male Broadcast Narrator (Authoritative, crisp, professional documentary narration)
* **Target Text:** "Artificial intelligence and speech technologies are transforming human communication worldwide."
* **🔴 Baseline Transcript:** "Artificial intelligence and speech technologies are transforming human communication worldwide."
* **🟡 Constant (0.85) Transcript:** "Artificial intelligence and speech technologies are transforming human communication worldwide."
* **🟢 Adaptive Transcript:** "Artificial intelligence and speech technologies are transforming human communication worldwide."

### Sample 4: Male Academic Lecture (Fast-paced, intellectual, analytical scientific lecture)
* **Target Text:** "Spectral analysis demonstrates a significant reduction in boundary transition turbulence."
* **🔴 Baseline Transcript:** "Spectral analysis demonstrates a significant reduction in boundary transition turbulence."
* **🟡 Constant (0.85) Transcript:** "Spectral analysis demonstrates a significant reduction in boundary transition turbulence."
* **🟢 Adaptive Transcript:** "Spectral analysis demonstrates a significant reduction in boundary transition turbulence."

### Sample 5: Deep Dramatic Cinematic Voice (Dark, gravelly, intense theatrical male voice)
* **Target Text:** "Why so serious? We have eliminated the boundary collision completely."
* **🔴 Baseline Transcript:** "Why so serious? We have eliminated the boundary collision completely."
* **🟡 Constant (0.85) Transcript:** "Why so serious? We have eliminated the boundary collision completely."
* **🟢 Adaptive Transcript:** "Why so serious? We have eliminated the boundary collision completely."

