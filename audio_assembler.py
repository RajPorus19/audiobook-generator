"""
Audio Assembler — stitches per-chunk audio arrays into a seamless audiobook.

Pipeline:
  1. Normalise loudness of each chunk independently to –16 LUFS (broadcast standard).
  2. Insert silence pauses between chunks:
       - Sentence boundary → 150–250 ms (randomised slightly for naturalness)
       - Paragraph boundary → 400–600 ms (randomised slightly)
  3. Apply a gentle low-pass filter (scipy Butterworth) across the final
     concatenated array to smooth any spectral discontinuities at boundaries.
  4. Export to WAV and/or MP3 via pydub.

–16 LUFS is the EBU R128 / ATSC A/85 standard for spoken-word audio, widely
used by audiobook publishers (Audible ACX standard: –18 to –23 LUFS peak).
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal  # type: ignore[import]

from text_processor import ChunkType

logger = logging.getLogger(__name__)

try:
    import pyloudnorm as pyln  # type: ignore[import]
    _PYLOUDNORM_AVAILABLE = True
except ImportError:
    _PYLOUDNORM_AVAILABLE = False
    logger.warning(
        "pyloudnorm not installed — loudness normalisation will be skipped. "
        "Install with: pip install pyloudnorm"
    )

# ---------------------------------------------------------------------------
# Pause durations (in seconds)
# ---------------------------------------------------------------------------
_SENTENCE_PAUSE_MIN: float = 0.15
_SENTENCE_PAUSE_MAX: float = 0.25
_PARAGRAPH_PAUSE_MIN: float = 0.40
_PARAGRAPH_PAUSE_MAX: float = 0.60

# ---------------------------------------------------------------------------
# Loudness target
# ---------------------------------------------------------------------------
_TARGET_LUFS: float = -16.0

# ---------------------------------------------------------------------------
# Low-pass filter — applied to the final output to smooth chunk boundaries.
# 8 kHz cutoff at 24 kHz sample rate suppresses synthesis artefacts while
# preserving all speech-relevant frequencies (human voice: 100 Hz – 7 kHz).
# ---------------------------------------------------------------------------
_LP_CUTOFF_HZ: float = 8000.0
_LP_ORDER: int = 4


class AudioAssembler:
    """Assembles a list of audio arrays into a single normalised audio file.

    Example::

        assembler = AudioAssembler(sample_rate=24000)
        final_wav = assembler.assemble(audio_chunks, chunk_types)
        assembler.export(final_wav, "output.mp3")
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        """Initialise the assembler.

        Args:
            sample_rate: Sample rate shared by all input arrays (Hz).
        """
        self._sr = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        chunks: List[np.ndarray],
        chunk_types: List[ChunkType],
    ) -> np.ndarray:
        """Normalise, insert pauses, concatenate, and filter.

        Args:
            chunks: List of float32 NumPy arrays, one per text chunk.
            chunk_types: Corresponding :class:`~text_processor.ChunkType` for
                each chunk — determines the silence duration inserted after it.

        Returns:
            A single float32 NumPy array representing the complete audio.

        Raises:
            ValueError: If ``chunks`` and ``chunk_types`` differ in length, or
                if no chunks are provided.
        """
        if not chunks:
            raise ValueError("No audio chunks provided.")
        if len(chunks) != len(chunk_types):
            raise ValueError(
                f"chunks ({len(chunks)}) and chunk_types ({len(chunk_types)}) "
                "must have the same length."
            )

        logger.info("Assembling %d audio chunks …", len(chunks))

        normalised = self._normalise_chunks(chunks)
        parts: List[np.ndarray] = []

        for idx, (audio, ctype) in enumerate(zip(normalised, chunk_types)):
            parts.append(audio)
            if idx < len(normalised) - 1:
                pause = self._make_pause(ctype)
                parts.append(pause)

        combined = np.concatenate(parts, axis=0).astype(np.float32)
        combined = self._apply_lowpass(combined)
        logger.info(
            "Assembly complete — %.2f seconds of audio.",
            len(combined) / self._sr,
        )
        return combined

    def export(
        self,
        audio: np.ndarray,
        output_path: str,
        export_wav: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """Write the assembled audio to MP3 and optionally WAV.

        Args:
            audio: Float32 NumPy array of audio samples.
            output_path: Destination file path (should end in ``.mp3``).
            export_wav: If True, also write a ``.wav`` file alongside the MP3.

        Returns:
            Tuple of (mp3_path, wav_path_or_None).

        Raises:
            ImportError: If pydub is not installed.
        """
        try:
            from pydub import AudioSegment  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pydub is required for export. Install with: pip install pydub"
            ) from exc

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 [−1, 1] to int16 PCM.
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)

        segment = AudioSegment(
            data=pcm_int16.tobytes(),
            sample_width=2,  # 16-bit
            frame_rate=self._sr,
            channels=1,
        )

        mp3_path = str(output.with_suffix(".mp3"))
        segment.export(mp3_path, format="mp3", bitrate="192k")
        logger.info("Exported MP3: %s", mp3_path)

        wav_path: Optional[str] = None
        if export_wav:
            wav_path = str(output.with_suffix(".wav"))
            segment.export(wav_path, format="wav")
            logger.info("Exported WAV: %s", wav_path)

        return mp3_path, wav_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_chunks(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        """Normalise each chunk to ``_TARGET_LUFS``.

        Falls back to peak normalisation if pyloudnorm is unavailable or if
        the chunk is too short for integrated loudness measurement (< 0.4 s).

        Args:
            chunks: List of float32 audio arrays.

        Returns:
            List of loudness-normalised float32 arrays.
        """
        normalised: List[np.ndarray] = []
        meter = pyln.Meter(self._sr) if _PYLOUDNORM_AVAILABLE else None

        for idx, chunk in enumerate(chunks):
            if meter is not None and len(chunk) / self._sr >= 0.4:
                try:
                    loudness = meter.integrated_loudness(chunk.astype(np.float64))
                    if np.isfinite(loudness) and loudness > -70.0:
                        norm = pyln.normalize.loudness(
                            chunk.astype(np.float64),
                            loudness,
                            _TARGET_LUFS,
                        )
                        normalised.append(norm.astype(np.float32))
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Chunk %d loudness normalisation failed: %s. "
                        "Falling back to peak normalisation.",
                        idx,
                        exc,
                    )
            # Peak normalise as fallback.
            peak = np.max(np.abs(chunk))
            if peak > 1e-6:
                normalised.append(chunk / peak * 0.9)
            else:
                normalised.append(chunk)

        return normalised

    def _make_pause(self, chunk_type: ChunkType) -> np.ndarray:
        """Create a silence array for the given boundary type.

        Adds a small random jitter to the pause duration to prevent the
        assembled audio from sounding metronomic.

        Args:
            chunk_type: Whether the preceding chunk ended a sentence or
                a paragraph.

        Returns:
            Float32 zero array of the appropriate length.
        """
        if chunk_type == ChunkType.PARAGRAPH_END:
            duration = random.uniform(_PARAGRAPH_PAUSE_MIN, _PARAGRAPH_PAUSE_MAX)
        else:
            duration = random.uniform(_SENTENCE_PAUSE_MIN, _SENTENCE_PAUSE_MAX)
        num_samples = int(self._sr * duration)
        return np.zeros(num_samples, dtype=np.float32)

    def _apply_lowpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply a Butterworth low-pass filter to smooth boundary artefacts.

        Args:
            audio: Full concatenated float32 audio array.

        Returns:
            Filtered float32 audio array.
        """
        nyquist = self._sr / 2.0
        cutoff_norm = min(_LP_CUTOFF_HZ / nyquist, 0.99)
        b, a = signal.butter(_LP_ORDER, cutoff_norm, btype="low", analog=False)
        filtered = signal.sosfilt(
            signal.butter(_LP_ORDER, cutoff_norm, btype="low", output="sos"),
            audio,
        )
        return filtered.astype(np.float32)
