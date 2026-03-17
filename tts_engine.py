"""
TTS Engine — Core wrapper around Qwen3-TTS.

Why Qwen3-TTS over alternatives:
  - gTTS / pyttsx3: robotic quality, no voice cloning, online dependency for gTTS.
  - Amazon Polly / ElevenLabs: cloud API — requires network, costs money, data leaves device.
  - Bark (suno-ai): high quality but extremely slow, no deterministic voice cloning.
  - Coqui TTS (XTTS v2): outdated, less natural prosody.
  - Qwen3-TTS: fully local, Apache-2.0, state-of-the-art naturalness, voice cloning from 3s.

Model variants (all under Qwen/Qwen3-TTS-12Hz-*):
  - 1.7B-CustomVoice : predefined speakers + style instructions (default, no ref audio needed).
  - 1.7B-Base        : voice cloning from a short reference WAV (~3 s is enough).
  - 0.6B-CustomVoice : lighter variant for low-VRAM setups.

The 1.7B-Base model is used automatically when a speaker_wav is supplied.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE: int = 24_000
_TORCH_SEED: int = 42

# Default voice used when no reference WAV is provided.
_DEFAULT_SPEAKER: str = "aiden"
_DEFAULT_INSTRUCTION: str = "Warm, natural, storytelling tone. Calm and engaging."

# HuggingFace model IDs
_MODEL_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_MODEL_VOICE_CLONE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


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
    """Wrapper around Qwen3-TTS for audiobook synthesis.

    Two operating modes:
      - CustomVoice (default): uses a named built-in speaker with an optional
        style instruction.  No reference audio required.
      - Voice cloning: when ``speaker_wav`` is supplied, loads the Base model
        and conditions generation on the reference clip every call.

    The model is loaded exactly once at construction time.

    Attributes:
        sample_rate: Output sample rate (always 24 000 Hz for Qwen3-TTS).
    """

    def __init__(
        self,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        speaker: str = _DEFAULT_SPEAKER,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        """Initialise the TTS engine.

        Args:
            speaker_wav: Path to a ≥3 second reference WAV for voice cloning.
                Optional — if omitted the built-in ``speaker`` voice is used.
            language: BCP-47 language code, e.g. 'en', 'fr', 'de'.
                Informational only; Qwen3-TTS is multilingual by default.
            speed: Speech-rate multiplier in [0.7, 1.3].
            speaker: Named speaker for CustomVoice mode (ignored when
                ``speaker_wav`` is provided).
            instruction: Natural-language style prompt for CustomVoice mode.

        Raises:
            FileNotFoundError: If ``speaker_wav`` is given but does not exist.
            RuntimeError: If the qwen-tts package is not installed.
        """
        random.seed(_TORCH_SEED)
        torch.manual_seed(_TORCH_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_TORCH_SEED)

        self._language = language
        self._speed = max(0.7, min(1.3, speed))
        self._device = _detect_device()
        self._speaker_wav: Optional[str] = None
        self._speaker = speaker
        self._instruction = instruction

        if speaker_wav is not None:
            path = Path(speaker_wav)
            if not path.exists():
                raise FileNotFoundError(f"speaker_wav not found: {speaker_wav}")
            self._speaker_wav = str(path.resolve())

        self._model = self._load_model()
        self.sample_rate: int = _SAMPLE_RATE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):  # type: ignore[return]
        """Load the appropriate Qwen3-TTS model variant.

        Returns:
            A loaded ``Qwen3TTSModel`` instance.

        Raises:
            RuntimeError: If the qwen-tts package is not installed.
        """
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "qwen-tts is not installed. Run: pip install qwen-tts soundfile"
            ) from exc

        model_id = _MODEL_VOICE_CLONE if self._speaker_wav else _MODEL_CUSTOM_VOICE

        # bfloat16 is the recommended dtype for CUDA; float32 for CPU/MPS.
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        # device_map format expected by Qwen3TTSModel.from_pretrained
        if self._device == "cuda":
            device_map = "cuda:0"
        elif self._device == "mps":
            device_map = "mps:0"
        else:
            device_map = "cpu"

        logger.info("Loading Qwen3-TTS model: %s  (device=%s)", model_id, device_map)
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
        )
        logger.info("Qwen3-TTS model loaded successfully.")
        return model

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

        if self._speaker_wav:
            wavs, fs = self._model.generate_voice_clone(
                text=text,
                ref_audio=self._speaker_wav,
                x_vector_only_mode=True,
            )
        else:
            wavs, fs = self._model.generate_custom_voice(
                text=text,
                speaker=self._speaker,
                instruct=self._instruction,
            )

        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
        return np.array(audio, dtype=np.float32)
