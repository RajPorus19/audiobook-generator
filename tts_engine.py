"""
TTS Engine — Core wrapper around Coqui TTS (XTTS v2).

Why Coqui TTS over alternatives:
  - gTTS / pyttsx3: robotic quality, no voice cloning, online dependency for gTTS.
  - Amazon Polly / ElevenLabs: cloud API — requires network, costs money, data leaves device.
  - Bark (suno-ai): high quality but extremely slow, no deterministic voice cloning.
  - Coqui TTS: fully local, Apache-2.0, state-of-the-art naturalness, supports XTTS v2.

Why XTTS v2 specifically:
  - Neural codec language model conditioned on a speaker reference — voice stays identical
    across every chunk because the conditioning embedding is fixed.
  - 17-language support, cross-lingual voice cloning, 24 kHz output.
  - Temperature / repetition-penalty controls allow reproducible, consistent prosody.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation constants — MUST NOT be changed per-chunk to guarantee
# voice and prosody consistency across the entire audiobook.
# ---------------------------------------------------------------------------
_TEMPERATURE: float = 0.65
_REPETITION_PENALTY: float = 2.5
_LENGTH_PENALTY: float = 1.0
_TORCH_SEED: int = 42


def _detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)
    return device


class TTSEngine:
    """Wrapper around Coqui TTS XTTS v2 for audiobook synthesis.

    The model is loaded exactly once at construction time. A speaker embedding
    is extracted from the reference WAV once and reused for every chunk, which
    is the key mechanism that guarantees a consistent voice throughout.

    Attributes:
        sample_rate: Output sample rate reported by the loaded model.
    """

    # Primary model — best quality, cross-lingual voice cloning.
    _PRIMARY_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    # Fallback if primary cannot be loaded (e.g. first-run download fails).
    _FALLBACK_MODEL = "tts_models/en/ljspeech/vits"

    def __init__(
        self,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> None:
        """Initialise the TTS engine and optionally extract a speaker embedding.

        Args:
            speaker_wav: Path to a 6–30 second clean-speech WAV file used for
                voice cloning.  Optional — if omitted a default XTTS speaker is
                used.
            language: BCP-47 language code, e.g. 'en', 'fr', 'de'.
            speed: Speech-rate multiplier in [0.7, 1.3].

        Raises:
            FileNotFoundError: If speaker_wav is given but does not exist.
            RuntimeError: If neither the primary nor the fallback model loads.
        """
        # Deterministic seed — set before any model weights are loaded.
        random.seed(_TORCH_SEED)
        torch.manual_seed(_TORCH_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_TORCH_SEED)

        self._language = language
        self._speed = max(0.7, min(1.3, speed))
        self._device = _detect_device()
        self._speaker_wav: Optional[str] = None
        self._gpt_cond_latent: Optional[torch.Tensor] = None
        self._speaker_embedding: Optional[torch.Tensor] = None
        self._use_xtts: bool = True

        if speaker_wav is not None:
            path = Path(speaker_wav)
            if not path.exists():
                raise FileNotFoundError(f"speaker_wav not found: {speaker_wav}")
            self._speaker_wav = str(path.resolve())

        self._tts = self._load_model()
        self.sample_rate: int = self._tts.synthesizer.output_sample_rate

        if self._use_xtts and self._speaker_wav:
            self._extract_speaker_embedding()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):  # type: ignore[return]
        """Load the Coqui TTS model, falling back gracefully on error.

        Returns:
            A loaded ``TTS`` instance.

        Raises:
            RuntimeError: If both primary and fallback loading fail.
        """
        # Import here so that the rest of the module is importable even if TTS
        # is not installed (useful for unit-testing the text / audio layers).
        try:
            from TTS.api import TTS  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS is not installed. Run: pip install TTS"
            ) from exc

        gpu = self._device == "cuda"

        logger.info("Loading primary TTS model: %s", self._PRIMARY_MODEL)
        try:
            model = TTS(model_name=self._PRIMARY_MODEL, progress_bar=True, gpu=gpu)
            self._use_xtts = True
            logger.info("XTTS v2 loaded successfully.")
            return model
        except Exception as primary_exc:  # noqa: BLE001
            logger.warning(
                "Primary model failed (%s). Falling back to %s.",
                primary_exc,
                self._FALLBACK_MODEL,
            )

        try:
            model = TTS(model_name=self._FALLBACK_MODEL, progress_bar=True, gpu=gpu)
            self._use_xtts = False
            logger.info("Fallback model loaded successfully.")
            return model
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Both TTS models failed to load. "
                f"Primary: {primary_exc}. Fallback: {fallback_exc}."
            ) from fallback_exc

    def _extract_speaker_embedding(self) -> None:
        """Extract and cache the GPT conditioning latent and speaker embedding.

        This is called once at startup. The cached tensors are reused for every
        ``synthesize`` call to guarantee voice consistency.
        """
        logger.info(
            "Extracting speaker embedding from: %s", self._speaker_wav
        )
        # Access the underlying XTTS model through the TTS wrapper.
        xtts_model = self._tts.synthesizer.tts_model  # type: ignore[attr-defined]
        (
            self._gpt_cond_latent,
            self._speaker_embedding,
        ) = xtts_model.get_conditioning_latents(
            audio_path=[self._speaker_wav],
        )
        logger.info("Speaker embedding extracted and cached.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize a text chunk into raw audio samples.

        Args:
            text: The text to synthesize.  Should be a single sentence or
                short paragraph — see ``TextProcessor`` for chunking.

        Returns:
            A 1-D float32 NumPy array of audio samples at ``self.sample_rate``.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        if self._use_xtts:
            return self._synthesize_xtts(text)
        return self._synthesize_fallback(text)

    def _default_speaker(self) -> str:
        """Return the first available built-in XTTS speaker name.

        Returns:
            A speaker name string suitable for passing to ``tts.tts(speaker=...)``.

        Raises:
            RuntimeError: If the model reports no speakers.
        """
        speakers = self._tts.speakers  # type: ignore[attr-defined]
        if not speakers:
            raise RuntimeError("XTTS v2 reported no built-in speakers.")
        return speakers[0]

    def _synthesize_xtts(self, text: str) -> np.ndarray:
        """Synthesize using XTTS v2 with fixed speaker conditioning.

        Args:
            text: Input text chunk.

        Returns:
            Float32 NumPy array of audio samples.
        """
        xtts_model = self._tts.synthesizer.tts_model  # type: ignore[attr-defined]

        if self._gpt_cond_latent is not None and self._speaker_embedding is not None:
            # Use the pre-extracted, fixed speaker embedding.
            out = xtts_model.inference(
                text=text,
                language=self._language,
                gpt_cond_latent=self._gpt_cond_latent,
                speaker_embedding=self._speaker_embedding,
                temperature=_TEMPERATURE,
                repetition_penalty=_REPETITION_PENALTY,
                length_penalty=_LENGTH_PENALTY,
                speed=self._speed,
            )
        else:
            # No reference WAV — use the first available built-in speaker.
            # XTTS v2 is a multi-speaker model; a speaker name is required
            # when no speaker_wav / embedding is provided.
            speaker = self._default_speaker()
            logger.info("No speaker_wav provided; using built-in speaker: %s", speaker)
            out = self._tts.tts(
                text=text,
                speaker=speaker,
                language=self._language,
                speed=self._speed,
            )
            if isinstance(out, list):
                return np.array(out, dtype=np.float32)
            return np.array(out, dtype=np.float32)

        wav = out.get("wav") if isinstance(out, dict) else out
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        return np.array(wav, dtype=np.float32)

    def _synthesize_fallback(self, text: str) -> np.ndarray:
        """Synthesize using the Tacotron2-DDC fallback model.

        Args:
            text: Input text chunk.

        Returns:
            Float32 NumPy array of audio samples.
        """
        wav = self._tts.tts(text=text)
        return np.array(wav, dtype=np.float32)
