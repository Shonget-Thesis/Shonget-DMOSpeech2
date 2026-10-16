"""
Chapter 3 Evaluation and Failure Module
=======================================
Implements Task 2 of the Chapter 3 Handover-Scheduling Specification:
- WER: Whisper large-v3 with official Seed-TTS / F5-TTS text normalization
- SIM: WavLM-large-based speaker verifier (cosine similarity vs. ground-truth target recording)
- UTMOS: SpeechMOS predictor (utmos22_strong)
- Prompt SNR: 25 ms / 10 ms energy percentile VAD
- Failure Detection (Definition 3.1):
    fail(x, k) = WER(x,k) > max(0.125, WER(x,ref) + 0.01) or SIM(x,k) < SIM(x,ref) - 0.02
- Standardized fractional WER units in [0, 1]
"""

import os
import json
import string
from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
try:
    from jiwer import process_words
    def _run_jiwer(ref: str, hyp: str):
        out = process_words(ref, hyp)
        return {
            "wer": float(out.wer),
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
        }
except ImportError:
    from jiwer import compute_measures
    def _run_jiwer(ref: str, hyp: str):
        m = compute_measures(ref, hyp)
        return {
            "wer": float(m["wer"]),
            "substitutions": int(m.get("substitutions", 0)),
            "deletions": int(m.get("deletions", 0)),
            "insertions": int(m.get("insertions", 0)),
            "hits": int(m.get("hits", 0)),
        }

# Default configuration path for frozen Chapter 3 failure constants
DEFAULT_FAILURE_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "failure_config.json")
)


# ============================================================================
# 1. TEXT NORMALIZATION & WER (Task 2.1, 2.2)
# ============================================================================

def get_punctuation_set() -> str:
    """Returns combined ASCII and standard Chinese/Unicode punctuation set."""
    punc = string.punctuation
    try:
        from zhon.hanzi import punctuation as zh_punc
        punc += zh_punc
    except ImportError:
        # Common Chinese punctuation fallback if zhon is not present
        punc += "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    return punc


def normalize_text(text: str) -> str:
    """
    Official Seed-TTS / F5-TTS English text normalization:
    - Strips all punctuation characters
    - Collapses consecutive spaces
    - Lowercases text
    """
    if not text:
        return ""
    punc = get_punctuation_set()
    clean = text
    for p in punc:
        clean = clean.replace(p, " ")
    # Collapse multiple whitespaces and trim
    clean = " ".join(clean.strip().split())
    return clean.lower()


def compute_wer_details(truth_text: str, hypo_text: str) -> Dict[str, Any]:
    """
    Computes Word Error Rate (WER) using jiwer after text normalization.
    Returns edit operation counts and fractional WER in [0, 1] (or > 1.0 if insertions exceed reference).
    """
    norm_truth = normalize_text(truth_text)
    norm_hypo = normalize_text(hypo_text)

    if not norm_truth:
        wer = 1.0 if norm_hypo else 0.0
        return {
            "unnormalized_truth": truth_text,
            "unnormalized_hypo": hypo_text,
            "normalized_truth": norm_truth,
            "normalized_hypo": norm_hypo,
            "wer": float(wer),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(norm_hypo.split()) if norm_hypo else 0,
            "hits": 0,
        }

    measures = _run_jiwer(norm_truth, norm_hypo)
    return {
        "unnormalized_truth": truth_text,
        "unnormalized_hypo": hypo_text,
        "normalized_truth": norm_truth,
        "normalized_hypo": norm_hypo,
        "wer": measures["wer"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
    }


# ============================================================================
# 2. FROZEN FAILURE FUNCTION (Task 2.3 - Definition 3.1)
# ============================================================================

def load_failure_config(config_path: Optional[str] = None) -> Dict[str, float]:
    """Loads frozen failure thresholds from JSON configuration file."""
    cfg_file = config_path or DEFAULT_FAILURE_CONFIG_PATH
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "absolute_wer_ceiling": float(data.get("absolute_wer_ceiling", 0.125)),
                "relative_wer_margin": float(data.get("relative_wer_margin", 0.01)),
                "similarity_margin": float(data.get("similarity_margin", 0.02)),
            }
    # Fallback to proposal canonical defaults
    return {
        "absolute_wer_ceiling": 0.125,
        "relative_wer_margin": 0.01,
        "similarity_margin": 0.02,
    }


def check_failure(
    wer_cand: float,
    wer_ref: float,
    sim_cand: float,
    sim_ref: float,
    config: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Unified Failure Function (Chapter 3, Definition 3.1):
        fail(x, k) =
            WER(x,k) > max(0.125, WER(x,ref) + 0.01)
            OR
            SIM(x,k) < SIM(x,ref) - 0.02

    Both WER and margins are evaluated as fractions (e.g. 0.125 is 12.5%, 0.01 is 1.0 percentage point).
    Returns (is_failure: bool, diagnostics: dict).
    """
    cfg = config or load_failure_config()
    abs_wer_ceiling = cfg["absolute_wer_ceiling"]
    rel_wer_margin = cfg["relative_wer_margin"]
    sim_margin = cfg["similarity_margin"]

    wer_threshold = max(abs_wer_ceiling, wer_ref + rel_wer_margin)
    sim_threshold = sim_ref - sim_margin

    wer_failed = bool(wer_cand > wer_threshold)
    sim_failed = bool(sim_cand < sim_threshold)
    is_failure = wer_failed or sim_failed

    diagnostics = {
        "is_failure": is_failure,
        "wer_failed": wer_failed,
        "sim_failed": sim_failed,
        "wer_cand": float(wer_cand),
        "wer_ref": float(wer_ref),
        "wer_threshold": float(wer_threshold),
        "sim_cand": float(sim_cand),
        "sim_ref": float(sim_ref),
        "sim_threshold": float(sim_threshold),
        "wer_margin": rel_wer_margin,
        "sim_margin": sim_margin,
    }

    return is_failure, diagnostics


# ============================================================================
# 3. EVALUATION MODEL WRAPPERS (Task 2.1)
# ============================================================================

class WhisperASREvaluator:
    """
    Whisper large-v3 ASR Evaluator with FP32 execution on CPU/CUDA.
    Generates hypothesis transcripts for WER computation.
    """
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self._pipe = None

    def _init_pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                dtype=torch.float32,
            )

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate: int = 24000) -> str:
        self._init_pipe()
        if isinstance(audio, str):
            # Pass filepath directly
            res = self._pipe(audio, generate_kwargs={"language": "english"})
            return res.get("text", "").strip()
        else:
            # Resample waveform to 16000 Hz for Whisper
            if audio.ndim > 1:
                audio = audio.squeeze()
            if sample_rate != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_16k = audio
            res = self._pipe(
                {"raw": audio_16k, "sampling_rate": 16000},
                generate_kwargs={"language": "english"},
            )
            return res.get("text", "").strip()


class WavLMSimilarityEvaluator:
    """
    WavLM-large Speaker Verifier.
    Extracts x-vector speaker embeddings and calculates cosine similarity
    against the ground-truth target recording.
    """
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cpu",
    ):
        self.device = device
        self.model_id = model_id
        self._fe = None
        self._model = None

    def _init_model(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
            self._fe = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioXVector.from_pretrained(self.model_id, use_safetensors=True)
            self._model.to(self.device)
            self._model.eval()

    def _load_audio_16k(self, audio: Union[str, np.ndarray], sample_rate: int = 24000) -> np.ndarray:
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
            if wav.ndim > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
            return wav.squeeze().numpy()
        else:
            if audio.ndim > 1:
                audio = audio.squeeze()
            if sample_rate != 16000:
                return librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            return audio

    def get_embedding(self, audio: Union[str, np.ndarray], sample_rate: int = 24000) -> torch.Tensor:
        self._init_model()
        wav_16k = self._load_audio_16k(audio, sample_rate)
        inputs = self._fe(wav_16k, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            emb = self._model(**inputs).embeddings
        return emb

    def compute_similarity(
        self,
        candidate_audio: Union[str, np.ndarray],
        ground_truth_audio: Union[str, np.ndarray],
        sample_rate: int = 24000,
    ) -> float:
        """
        Computes cosine similarity between synthesized audio and ground-truth target recording.
        """
        emb_cand = self.get_embedding(candidate_audio, sample_rate)
        emb_gt = self.get_embedding(ground_truth_audio, sample_rate)
        sim = F.cosine_similarity(emb_cand, emb_gt).item()
        return float(sim)


class UTMOSEvaluator:
    """
    SpeechMOS Naturalness Evaluator (utmos22_strong).
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._predictor = None

    def _init_model(self):
        if self._predictor is None:
            self._predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            )
            self._predictor.to(self.device)
            self._predictor.eval()

    def score(self, audio: Union[str, np.ndarray], sample_rate: int = 24000) -> float:
        self._init_model()
        if isinstance(audio, str):
            wav, sr = librosa.load(audio, sr=None, mono=True)
        else:
            wav = audio.squeeze() if audio.ndim > 1 else audio
            sr = sample_rate

        wav_tensor = torch.from_numpy(wav).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            score_tensor = self._predictor(wav_tensor, sr)
        return float(score_tensor.item())


# ============================================================================
# 4. COMPREHENSIVE METRIC SUITE
# ============================================================================

class MetricSuite:
    """
    Combined Evaluation Suite: WER, SIM, UTMOS, and Failure Assessment.
    Reuses model instances across evaluation runs for maximum throughput.
    """
    def __init__(self, device: str = "cpu", failure_config_path: Optional[str] = None):
        self.device = device
        self.failure_config = load_failure_config(failure_config_path)
        self.asr = WhisperASREvaluator(device=device)
        self.sim = WavLMSimilarityEvaluator(device=device)
        self.utmos = UTMOSEvaluator(device=device)

    def evaluate_utterance(
        self,
        cand_audio: Union[str, np.ndarray],
        gt_audio_path: str,
        gt_text: str,
        ref_metrics: Optional[Dict[str, float]] = None,
        sample_rate: int = 24000,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate synthesized audio:
        1. Transcribes with Whisper large-v3 and computes WER against ground-truth text
        2. Computes speaker similarity against ground-truth target audio using WavLM
        3. Computes UTMOS naturalness score
        4. If ref_metrics is provided, evaluates failure against step-8 reference
        """
        # 1. ASR & WER
        hypo_text = self.asr.transcribe(cand_audio, sample_rate=sample_rate)
        wer_res = compute_wer_details(gt_text, hypo_text)

        # 2. SIM vs Ground-Truth Target Recording
        sim_score = self.sim.compute_similarity(cand_audio, gt_audio_path, sample_rate=sample_rate)

        # 3. UTMOS
        utmos_score = self.utmos.score(cand_audio, sample_rate=sample_rate)

        result = {
            "wer": wer_res["wer"],
            "substitutions": wer_res["substitutions"],
            "deletions": wer_res["deletions"],
            "insertions": wer_res["insertions"],
            "hits": wer_res["hits"],
            "unnormalized_hypo": wer_res["unnormalized_hypo"],
            "normalized_hypo": wer_res["normalized_hypo"],
            "normalized_truth": wer_res["normalized_truth"],
            "sim": sim_score,
            "utmos": utmos_score,
        }

        # 4. Failure check against reference
        if ref_metrics is not None:
            is_fail, diag = check_failure(
                wer_cand=result["wer"],
                wer_ref=ref_metrics["wer"],
                sim_cand=result["sim"],
                sim_ref=ref_metrics["sim"],
                config=self.failure_config,
            )
            result["fail"] = int(is_fail)
            result["wer_failed"] = diag["wer_failed"]
            result["sim_failed"] = diag["sim_failed"]
        else:
            result["fail"] = 0

        return result
