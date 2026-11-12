import io
import re
import uuid
import base64
import logging
import difflib
import secrets
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf
import av
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import whisper

from app.config import settings

logger = logging.getLogger(__name__)

# Sample pool of phonetically diverse, non-trivial challenge phrases
CHALLENGE_PHRASE_POOL = [
    "The blue sky was clear and bright this morning",
    "Silver rivers flow gracefully across the valley",
    "Nine quick foxes jumped over seven wooden fences",
    "Bright sunshine warms the cold mountain peaks",
    "Strong ocean waves crashed against the rocky shore",
    "Golden leaves fall gently in the autumn breeze",
    "Careful acoustic analysis ensures secure authentication",
    "Every spoken word carries distinct spectral features",
    "Clear speech enables accurate biometric verification",
    "Fresh challenges protect against replay attacks"
]


class AudioQualityError(ValueError):
    """Audio cannot be used reliably for voice biometric processing."""

    def __init__(self, message: str, quality_details: Dict[str, Any]):
        super().__init__(message)
        self.quality_details = quality_details

class VoiceEngine:
    _instance: Optional["VoiceEngine"] = None
    classifier: Optional[EncoderClassifier] = None
    whisper_model: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceEngine, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        """Initialize SpeechBrain ECAPA-TDNN and Whisper models with error handling."""
        logger.info("Initializing SpeechBrain ECAPA-TDNN engine...")
        try:
            model_dir = Path(settings.VOICE_MODEL_DIR)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Load ECAPA-TDNN with Windows-safe COPY strategy
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
                local_strategy=LocalStrategy.COPY
            )
            logger.info("SpeechBrain ECAPA-TDNN model loaded successfully.")
        except Exception as e:
            logger.warning("Could not pre-warm SpeechBrain model: %s. Will retry on demand.", e)
            self.classifier = None

        logger.info("Initializing Whisper model (%s)...", settings.WHISPER_MODEL_SIZE)
        try:
            whisper_dir = str(Path(settings.VOICE_MODEL_DIR) / "whisper")
            self.whisper_model = whisper.load_model(
                settings.WHISPER_MODEL_SIZE,
                download_root=whisper_dir
            )
            logger.info("Whisper (%s) model loaded successfully.", settings.WHISPER_MODEL_SIZE)
        except Exception as e:
            logger.warning("Could not pre-warm Whisper model: %s. Will retry on demand.", e)
            self.whisper_model = None

    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return (self.classifier is not None) and (self.whisper_model is not None)

    @staticmethod
    def decode_base64_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
        """Decode a base64 string or data URL (e.g., data:audio/webm;base64,...) into PCM."""
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio payload: {e}")
        return VoiceEngine.decode_audio_payload(audio_bytes)

    @staticmethod
    def decode_audio_payload(content: bytes, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Decode raw audio bytes into 16 kHz mono float32 array.
        Supports WAV, FLAC, OGG via soundfile, and WebM, MP3, AAC, MP4 audio via PyAV.
        """
        if not content or len(content) == 0:
            raise ValueError("Audio payload is empty (0 bytes received).")

        # First attempt: Try standard SoundFile (very fast for WAV/FLAC/OGG)
        try:
            with io.BytesIO(content) as bio:
                data, sr = sf.read(bio, dtype="float32")
                # Convert multichannel to mono
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                # Resample to 16 kHz if necessary
                target_sr = settings.VOICE_SAMPLE_RATE
                if sr != target_sr:
                    data = VoiceEngine._resample_audio(data, sr, target_sr)
                    sr = target_sr

                return data.astype(np.float32), sr
        except Exception:
            # Fallback to PyAV which supports WebM Opus, MP4, MP3, etc.
            pass

        # Second attempt: PyAV general media container decoder
        try:
            with io.BytesIO(content) as bio:
                container = av.open(bio)
                audio_streams = [s for s in container.streams if s.type == "audio"]
                if not audio_streams:
                    raise ValueError("No audio stream found in the provided media file.")

                stream = audio_streams[0]
                target_sr = settings.VOICE_SAMPLE_RATE
                resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)

                pcm_chunks = []
                for frame in container.decode(stream):
                    resampled = resampler.resample(frame)
                    for r in resampled:
                        pcm_chunks.append(r.to_ndarray().flatten())

                if not pcm_chunks:
                    raise ValueError("Audio stream contains no decodable audio frames.")

                audio_data = np.concatenate(pcm_chunks).astype(np.float32)
                return audio_data, target_sr
        except Exception as e:
            raise ValueError(f"Unsupported or corrupted audio format: {e}")

    @staticmethod
    def _resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample 1D audio array to target sample rate using PyAV or numpy interpolation."""
        if orig_sr == target_sr:
            return data
        # Use simple high-quality polyphase or linear interpolation
        duration = len(data) / orig_sr
        target_len = int(duration * target_sr)
        if target_len <= 0:
            return np.zeros(0, dtype=np.float32)
        indices = np.linspace(0, len(data) - 1, target_len)
        resampled = np.interp(indices, np.arange(len(data)), data)
        return resampled.astype(np.float32)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Normalize a 1-D embedding vector to unit length, returning a safe float32 array."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            raise ValueError("Extracted voice embedding is empty.")
        norm = np.linalg.norm(emb)
        if not np.isfinite(norm) or norm <= 1e-10:
            raise ValueError("Extracted voice embedding has near-zero magnitude.")
        return (emb / norm).astype(np.float32)

    @staticmethod
    def compute_audio_quality(audio: np.ndarray, sr: int = 16000) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate audio quality metrics:
        1. Duration (sec)
        2. RMS Energy
        3. Clipping Ratio (percentage of samples near +/- 1.0)
        4. Estimated SNR (dB)
        Returns a normalized quality score in [0.0, 1.0] and detailed metrics.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr <= 0 or not np.all(np.isfinite(audio)):
            reason = "Audio has an invalid sample rate or contains non-finite samples."
            return 0.0, {
                "duration_sec": 0.0,
                "sample_rate": max(0, sr),
                "snr_db": 0.0,
                "clipping_ratio": 0.0,
                "rms_energy": 0.0,
                "active_frame_ratio": 0.0,
                "quality_status": "UNACCEPTABLE",
                "quality_message": reason,
            }
        duration = len(audio) / float(sr)
        if duration <= 0:
            return 0.0, {
                "duration_sec": 0.0,
                "sample_rate": sr,
                "snr_db": 0.0,
                "clipping_ratio": 0.0,
                "rms_energy": 0.0,
                "active_frame_ratio": 0.0,
                "quality_status": "UNACCEPTABLE",
                "quality_message": "Audio contains no samples."
            }

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))

        # 2. Clipping Ratio (samples above 0.98 in absolute value)
        clipping_count = int(np.sum(np.abs(audio) >= 0.98))
        clipping_ratio = float(clipping_count / len(audio))

        # 3. SNR estimation via frame energy percentiles
        frame_len = int(0.02 * sr)  # 20ms frames = 320 samples
        active_frame_ratio = 0.0
        if len(audio) >= frame_len:
            num_frames = len(audio) // frame_len
            frames = audio[:num_frames * frame_len].reshape(num_frames, frame_len)
            frame_energies = np.mean(frames**2, axis=1) + 1e-12
            active_frame_ratio = float(np.mean(np.sqrt(frame_energies) >= 0.005))
            # 15th percentile as noise floor, 85th percentile as speech energy
            noise_floor = np.percentile(frame_energies, 15)
            speech_energy = np.percentile(frame_energies, 85)
            snr_db = float(10.0 * np.log10(max(1e-6, speech_energy / (noise_floor + 1e-12))))
            snr_db = float(np.clip(snr_db, 0.0, 45.0))
        else:
            snr_db = 0.0

        # Quality scoring heuristic:
        # High SNR is good (target 20+ dB)
        snr_score = min(1.0, max(0.1, snr_db / 25.0))
        # Clipping penalty
        clip_score = max(0.0, 1.0 - (clipping_ratio * 15.0))
        # RMS energy score (audible speech typically has RMS 0.02 - 0.20)
        if rms < 0.005:
            rms_score = 0.2  # Near silence
        elif rms > 0.40:
            rms_score = 0.6  # Overdriven / saturated
        else:
            rms_score = 1.0

        quality = float(np.clip(0.45 * snr_score + 0.35 * clip_score + 0.20 * rms_score, 0.0, 1.0))

        reasons = []
        if rms < 0.005:
            reasons.append("Audio is silent or too quiet to process reliably.")
        if active_frame_ratio < 0.10:
            reasons.append("Too little of the recording contains audible speech.")
        if clipping_ratio > 0.20:
            reasons.append("Audio is excessively clipped or distorted.")

        quality_details = {
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "snr_db": round(snr_db, 2),
            "clipping_ratio": round(clipping_ratio, 4),
            "rms_energy": round(rms, 4),
            "active_frame_ratio": round(active_frame_ratio, 4),
            "quality_status": "UNACCEPTABLE" if reasons else "ACCEPTABLE",
            "quality_message": " ".join(reasons) if reasons else "Audio quality is acceptable."
        }

        return round(quality, 4), quality_details

    def extract_voice_embedding(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Validate duration and audio quality, extract a 192-d voice embedding.

        Extracts only a genuine SpeechBrain ECAPA-TDNN embedding. It never substitutes a synthetic fallback
        vector, because that would make biometric identity scores meaningless.
        """
        if sr <= 0:
            raise ValueError("Audio sample rate must be greater than zero.")
        duration = len(audio) / float(sr)
        if duration < settings.VOICE_MIN_DURATION_SEC:
            raise ValueError(
                f"Audio sample too short ({duration:.2f}s). "
                f"Minimum required duration is {settings.VOICE_MIN_DURATION_SEC:.1f} seconds."
            )
        if duration > settings.VOICE_MAX_DURATION_SEC:
            raise ValueError(
                f"Audio sample duration ({duration:.2f}s) exceeds maximum allowed "
                f"limit of {settings.VOICE_MAX_DURATION_SEC:.1f} seconds."
            )

        quality_score, quality_details = self.compute_audio_quality(audio, sr)
        if quality_details["quality_status"] != "ACCEPTABLE":
            raise AudioQualityError(quality_details["quality_message"], quality_details)

        if self.classifier is None:
            raise RuntimeError("SpeechBrain ECAPA-TDNN model is unavailable; no voice template was generated.")

        try:
            signal = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)
            with torch.no_grad():
                raw = self.classifier.encode_batch(signal)
                if isinstance(raw, (list, tuple)):
                    raw = raw[0]
                emb = np.asarray(raw.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"SpeechBrain ECAPA-TDNN inference failed: {exc}") from exc

        if emb.size != 192:
            raise RuntimeError(f"ECAPA-TDNN returned {emb.size} values; expected exactly 192.")
        if not np.all(np.isfinite(emb)):
            raise RuntimeError("ECAPA-TDNN returned a non-finite embedding.")
        try:
            emb = self._normalize_embedding(emb)
        except ValueError as exc:
            raise RuntimeError(f"ECAPA-TDNN returned an invalid embedding: {exc}") from exc

        return emb, quality_score, quality_details

    def transcribe_audio(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcribe spoken audio using OpenAI Whisper."""
        if self.whisper_model is None:
            self._initialize_models()
        if self.whisper_model is None:
            raise RuntimeError("Whisper transcription model is not available.")

        # Whisper expects float32 audio sampled at 16 kHz in range [-1.0, 1.0]
        audio_float = audio.astype(np.float32)
        # Avoid zero or near-silent input crash
        if len(audio_float) == 0 or np.max(np.abs(audio_float)) < 1e-4:
            return ""

        result = self.whisper_model.transcribe(
            audio_float,
            fp16=False,
            language="en"
        )
        return result.get("text", "").strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
        text = text.lower()
        # Remove all punctuation except alphanumeric and space
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def verify_challenge_phrase(cls, transcription: str, expected_phrase: str) -> Tuple[float, bool]:
        """
        Compare Whisper transcription against expected challenge phrase.
        Uses combined character-sequence similarity and token recall.
        Returns:
            phrase_similarity (float): [0.0, 1.0]
            phrase_matched (bool): True if similarity >= CHALLENGE_MATCH_THRESHOLD
        """
        norm_trans = cls.normalize_text(transcription)
        norm_exp = cls.normalize_text(expected_phrase)

        if not norm_exp:
            return 1.0, True
        if not norm_trans:
            return 0.0, False

        # 1. SequenceMatcher ratio (character-level Levenshtein-like alignment)
        seq_ratio = difflib.SequenceMatcher(None, norm_trans, norm_exp).ratio()

        # 2. Token recall (fraction of expected words recognized in transcription)
        exp_words = norm_exp.split()
        trans_words = set(norm_trans.split())
        matched_words = sum(1 for w in exp_words if w in trans_words)
        token_recall = matched_words / max(1, len(exp_words))

        # Balanced score
        similarity = 0.50 * seq_ratio + 0.50 * token_recall
        is_match = similarity >= settings.CHALLENGE_MATCH_THRESHOLD

        return round(float(similarity), 4), bool(is_match)

    @staticmethod
    def generate_random_challenge() -> Dict[str, Any]:
        """Generate a fresh random challenge phrase with unique nonce."""
        idx = secrets.randbelow(len(CHALLENGE_PHRASE_POOL))
        phrase = CHALLENGE_PHRASE_POOL[idx]
        return {
            "challenge_id": str(uuid.uuid4()),
            "challenge_phrase": phrase,
            "expires_in_seconds": 60
        }

# Global singleton helper
voice_engine = VoiceEngine()
