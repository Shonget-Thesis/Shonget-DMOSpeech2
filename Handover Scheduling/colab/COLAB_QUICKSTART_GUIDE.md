# Google Colab GPU Quickstart Guide (Turnkey & High-Speed)

This guide shows you how to run your **DMOSpeech 2 Adaptive Handover Scheduling** experiments in **Google Colab** using your **Google AI Pro / Colab Pro** GPU.

---

## Why Run on Colab GPU?
- **Speed:** What took **9 seconds per utterance on your laptop CPU** runs in **~0.15–0.25 seconds on an A100 or T4 GPU**.
- **Full Benchmark in Minutes:** Running 200 speakers (400 pairs / 1,600 candidate evaluations) completes in **~5 minutes on GPU** instead of days on CPU.
- **Expanded Scope:** You can easily run hundreds of additional speakers without draining your laptop's battery.

---

## 3 Easy Steps to Run in Google Colab:

### Step 1: Open the Notebook in Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload Notebook** and select:
   [`Handover Scheduling/colab/DMOSpeech2_Adaptive_Handover_Colab.ipynb`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/colab/DMOSpeech2_Adaptive_Handover_Colab.ipynb)
   *(Or click the **GitHub** tab in Colab, search for `Shonget-Thesis/Shonget-DMOSpeech2`, and open `DMOSpeech2_Adaptive_Handover_Colab.ipynb`).*

### Step 2: Set Runtime to GPU
1. In Colab's top menu, click **Runtime > Change runtime type**.
2. Select **T4 GPU** or **A100 GPU** (available with your Colab Pro / AI Pro account).
3. Click **Save**.

### Step 3: Click "Run All" (or execute cells one by one)
The notebook will automatically:
1. Verify the NVIDIA GPU (`!nvidia-smi`).
2. Clone your GitHub repository `https://github.com/Shonget-Thesis/Shonget-DMOSpeech2.git`.
3. Install all audio packages (`vocos`, `speechbrain`, `transformers`, etc.).
4. Download the model weights directly from HuggingFace (`model_85000.pt` and `model_1500.pt`).
5. Run the high-speed GPU benchmark:
   - Run a quick 10-speaker test (~20 seconds):
     ```bash
     !python "Handover Scheduling/colab/run_gpu_expanded_benchmark.py" --device cuda --scope test200 --n_speakers 10
     ```
   - Or run the full 200 held-out test speakers (~5 minutes):
     ```bash
     !python "Handover Scheduling/colab/run_gpu_expanded_benchmark.py" --device cuda --scope test200
     ```
   - Or run the **Expanded Scope** across all speakers in the dataset:
     ```bash
     !python "Handover Scheduling/colab/run_gpu_expanded_benchmark.py" --device cuda --scope expanded
     ```
6. Display the metrics summary table and let you listen to the generated speech right inside the Colab notebook!

---

## Files Included in this Repository:
- [`colab/DMOSpeech2_Adaptive_Handover_Colab.ipynb`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/colab/DMOSpeech2_Adaptive_Handover_Colab.ipynb): Turnkey Colab notebook.
- [`colab/run_gpu_expanded_benchmark.py`](file:///c:/Users/Rhenel%20Jhon%20Sajol/Desktop/TTS/DMOSpeech%202/Handover%20Scheduling/colab/run_gpu_expanded_benchmark.py): High-speed CUDA script for running batch evaluations.
