"""
Aligner — forced alignment of audio to transcript using WhisperX.

Why WhisperX over alternatives:
  - aeneas: requires Festival TTS installed system-wide; fragile on modern Linux.
  - gentle: unmaintained, requires Kaldi build, limited Python 3 support.
  - Montreal Forced Aligner (MFA): heavy Java + Kaldi dependency, complex setup,
    not pip-installable.
  - Raw Whisper timestamps: segment-level only, ±500ms accuracy — insufficient for
    smooth per-sentence highlight sync.
  - WhisperX: pip-installable, CUDA-accelerated, wav2vec2 forced alignment,
    20–40ms word-level accuracy, actively maintained.

Alignment strategy:
  WhisperX does NOT re-transcribe from scratch. It first auto-transcribes the audio
  to get approximate segment boundaries, then applies wav2vec2 forced alignment within
  those segments to pin each word to its exact audio position. Word-level accuracy
  is typically 20–40ms — sufficient for frame-accurate video sync at 30 FPS (33ms/frame).

  After alignment, a sequential word-matching algorithm maps transcribed word
  timestamps back to the original processed sentences.  Because the audio was
  synthesised directly from the processed text (abbreviations expanded, numbers
  spelled out), the transcribed words should closely mirror the source, making
  sequential matching highly reliable.
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Gap added to the last word's end time to form the sentence's deactivation time.
_END_BUFFER_MS: int = 80


@dataclass
class SentenceTimestamp:
    """Timing entry for a single sentence in the narration.

    Attributes:
        index: Zero-based sentence index in the full text.
        text: The processed sentence string (same form used for TTS synthesis).
        start_ms: Millisecond timestamp when the first word of the sentence begins.
        end_ms: Millisecond timestamp when the last word ends, plus an 80ms buffer.
    """

    index: int
    text: str
    start_ms: int
    end_ms: int


def _cudnn8_available() -> bool:
    """Check whether the cuDNN 8 shared library required by pyannote.audio is present.

    pyannote.audio's speechbrain VAD model links against ``libcudnn_ops_infer.so.8``
    (cuDNN 8.x).  Systems with cuDNN 9.x (CUDA 12.8+) no longer ship that file,
    causing a hard crash (SIGABRT) that Python cannot catch.  Probing the library
    with ctypes before committing to CUDA prevents the crash.

    Returns:
        True if ``libcudnn_ops_infer.so.8`` is loadable, False otherwise.
    """
    import ctypes
    for lib_name in ("libcudnn_ops_infer.so.8", "libcudnn_ops_infer.so"):
        try:
            ctypes.CDLL(lib_name)
            return True
        except OSError:
            continue
    return False


def _device_for_whisperx() -> str:
    """Detect the best available device for WhisperX inference.

    Returns 'cuda' only when both CUDA and cuDNN 8 are available — pyannote.audio's
    VAD model hard-crashes on cuDNN 9-only systems.  Falls back to 'cpu' otherwise.

    Returns:
        Device string: 'cuda' or 'cpu'.
    """
    try:
        import torch  # type: ignore[import]

        if torch.cuda.is_available():
            if _cudnn8_available():
                logger.info("WhisperX will use CUDA (cuDNN 8 confirmed).")
                return "cuda"
            logger.warning(
                "CUDA is available but libcudnn_ops_infer.so.8 was not found. "
                "pyannote.audio requires cuDNN 8 — falling back to CPU. "
                "To use GPU, install cuDNN 8: pip install nvidia-cudnn-cu12==8.9.7.29"
            )
    except ImportError:
        pass
    logger.info("WhisperX will use CPU.")
    return "cpu"


def _normalize_word(word: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for fuzzy matching.

    Args:
        word: Raw word string (may contain punctuation, mixed case).

    Returns:
        Normalised word containing only a-z, 0-9.
    """
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _words_match(a: str, b: str, min_prefix: int = 4) -> bool:
    """Check whether two normalised words should be considered a match.

    Exact match is always accepted.  Prefix match (≥ ``min_prefix`` chars)
    handles compound words like 'fortytwo' vs 'forty' when TTS splits a
    num2words result differently than Whisper reassembles it.

    Args:
        a: First normalised word.
        b: Second normalised word.
        min_prefix: Minimum prefix length to accept a partial match.

    Returns:
        True if the words are considered equivalent.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= min_prefix and len(b) >= min_prefix:
        return a.startswith(b) or b.startswith(a)
    return False


class Aligner:
    """Aligns a narration audio file to its source transcript at sentence level.

    Usage::

        aligner = Aligner()
        timestamps = aligner.align("narration.mp3", "path/to/text.txt")

    Attributes:
        device: Compute device string used for WhisperX ('cuda' or 'cpu').
    """

    def __init__(self, device: str = "auto") -> None:
        """Initialise the aligner.

        Args:
            device: Compute device.  Pass 'auto' to auto-detect CUDA / CPU.
        """
        self.device: str = _device_for_whisperx() if device == "auto" else device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: str,
        transcript_text: str,
        language: str = "en",
    ) -> List[SentenceTimestamp]:
        """Run forced alignment and return per-sentence timestamps.

        Args:
            audio_path: Path to the narration audio file (MP3 or WAV).
            transcript_text: The full raw text that was fed to TTS.
            language: BCP-47 language code for the alignment model.

        Returns:
            Ordered list of :class:`SentenceTimestamp` objects, one per sentence.

        Raises:
            ImportError: If ``whisperx`` is not installed.
            FileNotFoundError: If ``audio_path`` does not exist.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        sentences = self._extract_sentences(transcript_text)
        logger.info("Aligning %d sentences …", len(sentences))

        word_segments = self._run_whisperx(audio_path, language)
        timestamps = self._map_words_to_sentences(word_segments, sentences)
        timestamps = self._interpolate_missing(timestamps)

        return timestamps

    # ------------------------------------------------------------------
    # Cache helpers (used by video_main.py)
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(audio_path: str, text_path: str) -> str:
        """Compute a SHA-256 cache key from the two input file paths.

        The key encodes *both* the paths and the file contents so that any
        change to either file invalidates the cache.

        Args:
            audio_path: Path to the audio file.
            text_path: Path to the text file.

        Returns:
            64-character hex digest.
        """
        # Version string — bump whenever the extraction or mapping logic changes
        # so that stale caches are automatically invalidated.
        _VERSION = b"v2-sentence-level"
        h = hashlib.sha256()
        h.update(_VERSION)
        h.update(audio_path.encode())
        h.update(text_path.encode())
        for path in (audio_path, text_path):
            try:
                h.update(Path(path).read_bytes())
            except OSError:
                pass
        return h.hexdigest()

    @staticmethod
    def save_cache(
        timestamps: List[SentenceTimestamp],
        cache_path: str,
        key: str,
    ) -> None:
        """Write alignment results to a JSON cache file.

        Args:
            timestamps: List of :class:`SentenceTimestamp` to serialise.
            cache_path: Destination ``.json`` file path.
            key: Cache key to embed in the file (for validation on load).
        """
        payload = {
            "version": 1,
            "key": key,
            "sentences": [asdict(ts) for ts in timestamps],
        }
        Path(cache_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Alignment cache saved: %s", cache_path)

    @staticmethod
    def load_cache(
        cache_path: str,
        expected_key: str,
    ) -> Optional[List[SentenceTimestamp]]:
        """Load alignment results from a JSON cache file if the key matches.

        Args:
            cache_path: Path to the ``.json`` cache file.
            expected_key: Cache key to verify against the stored key.

        Returns:
            List of :class:`SentenceTimestamp` objects, or ``None`` if the cache
            is missing, corrupt, or the key does not match.
        """
        path = Path(cache_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("key") != expected_key:
                logger.info("Cache key mismatch — re-running alignment.")
                return None
            return [SentenceTimestamp(**s) for s in payload["sentences"]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache load failed (%s) — re-running alignment.", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sentences(transcript_text: str) -> List[str]:
        """Split the transcript into individual sentences for video sync.

        Uses TextProcessor for text cleaning (unicode, abbreviations, numbers)
        but then splits at the individual sentence level — NOT into 250-token
        TTS chunks.  One sentence per timestamp entry is required so that the
        scroll target changes on every sentence activation.

        Args:
            transcript_text: Full raw input text.

        Returns:
            Ordered list of individual cleaned sentence strings.
        """
        import nltk  # noqa: PLC0415
        from text_processor import TextProcessor  # noqa: PLC0415

        processor = TextProcessor()
        # Apply the same cleaning pipeline as TTS (abbreviations, numbers, unicode).
        cleaned = processor._clean(transcript_text)  # noqa: SLF001

        # Split on paragraph breaks first, then sentence-tokenize each paragraph.
        paragraphs = [p.strip() for p in cleaned.replace("\n\n", "\n\n").split("\n\n") if p.strip()]
        sentences: List[str] = []
        for para in paragraphs:
            para_sents = nltk.sent_tokenize(para)
            sentences.extend(s.strip() for s in para_sents if s.strip())

        logger.info("Extracted %d individual sentences for alignment.", len(sentences))
        return sentences

    def _run_whisperx(
        self,
        audio_path: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Transcribe and align the audio, returning word-level timestamps.

        Args:
            audio_path: Path to the audio file.
            language: Language code for the alignment model.

        Returns:
            List of word-segment dicts, each with keys ``'word'``, ``'start'``,
            ``'end'`` (all times in seconds).

        Raises:
            ImportError: If ``whisperx`` is not installed.
        """
        # ── torchaudio compatibility patch ──────────────────────────────────
        # torchaudio 2.x uses lazy loading for its C++ extensions, so several
        # symbols (AudioMetaData, list_audio_backends, get_audio_backend,
        # set_audio_backend) may not exist at the top-level namespace until a
        # native function is called.  pyannote.audio (pulled in by whisperx)
        # probes these symbols at MODULE LOAD TIME, causing AttributeError.
        # Fix: force-initialise the extension, then stub any remaining gaps.
        import torchaudio as _ta  # type: ignore[import]

        # 1. Try to trigger the official lazy-init path.
        try:
            import torchaudio._extension as _taext
            if hasattr(_taext, "_init_extension"):
                _taext._init_extension()
        except Exception:
            pass

        # 2. PyTorch 2.6 changed torch.load to weights_only=True by default.
        #    pyannote.audio checkpoints contain omegaconf objects with arbitrary
        #    custom classes — enumerating them all is fragile.  Instead, patch
        #    torch.load to keep weights_only=False as the default for calls that
        #    do not explicitly set it.  The pyannote/whisperx model files are
        #    trusted local checkpoints so this is safe.
        import torch as _torch
        import functools as _functools
        if not getattr(_torch.load, "_patched_weights_only", False):
            _orig_load = _torch.load

            @_functools.wraps(_orig_load)
            def _patched_load(*args, **kwargs):
                # Force weights_only=False — lightning_fabric sets it to True
                # explicitly, so setdefault is insufficient.
                kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)

            _patched_load._patched_weights_only = True  # type: ignore[attr-defined]
            _torch.load = _patched_load  # type: ignore[assignment]

        # 3. Stub any attributes still missing after the init attempt.
        if not hasattr(_ta, "AudioMetaData"):
            from collections import namedtuple as _nt
            _ta.AudioMetaData = _nt(  # type: ignore[attr-defined]
                "AudioMetaData",
                ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            )
        if not hasattr(_ta, "list_audio_backends"):
            _ta.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        if not hasattr(_ta, "get_audio_backend"):
            _ta.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
        if not hasattr(_ta, "set_audio_backend"):
            _ta.set_audio_backend = lambda _b: None  # type: ignore[attr-defined]

        try:
            import whisperx  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "whisperx is not installed. Run: pip install whisperx"
            ) from exc

        compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info("Loading WhisperX large-v2 model (device=%s) …", self.device)
        model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type=compute_type,
        )

        logger.info("Transcribing audio …")
        result = model.transcribe(audio_path, batch_size=16 if self.device == "cuda" else 4)
        detected_lang = result.get("language", language)
        logger.info("Detected language: %s", detected_lang)

        logger.info("Loading wav2vec2 alignment model …")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=self.device,
        )

        logger.info("Running forced alignment …")
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            self.device,
            return_char_alignments=False,
        )

        word_segments = aligned.get("word_segments", [])
        logger.info("Extracted %d word-level timestamps.", len(word_segments))
        return word_segments

    def _map_words_to_sentences(
        self,
        word_segments: List[Dict[str, Any]],
        sentences: List[str],
    ) -> List[SentenceTimestamp]:
        """Map word-level timestamps to their parent sentences.

        Uses a sequential sliding-window matcher: for each ASR word, scan forward
        up to 12 positions in the original word list to find a match.  The slack
        absorbs minor transcription differences without losing track of position.

        Args:
            word_segments: Word-level output from WhisperX alignment.
            sentences: Processed sentence strings in order.

        Returns:
            List of :class:`SentenceTimestamp` with ``start_ms`` / ``end_ms``
            filled in for sentences that received at least one matched word.
            Sentences with no matches have their times left as 0 (callers should
            run :meth:`_interpolate_missing` after this).
        """
        # Build flat (sentence_idx, normalised_word) list from original text.
        orig_words: List[Tuple[int, str]] = []
        for sent_idx, sentence in enumerate(sentences):
            for word in sentence.split():
                norm = _normalize_word(word)
                if norm:
                    orig_words.append((sent_idx, norm))

        sent_starts: Dict[int, int] = {}
        sent_ends: Dict[int, int] = {}

        orig_ptr = 0
        for ws in word_segments:
            asr_norm = _normalize_word(ws.get("word", ""))
            if not asr_norm:
                continue

            start_ms = int(ws.get("start", 0.0) * 1000)
            end_ms = int(ws.get("end", 0.0) * 1000)

            # Scan ahead for a matching original word.
            matched = False
            for slack in range(12):
                if orig_ptr + slack >= len(orig_words):
                    break
                sent_idx, orig_norm = orig_words[orig_ptr + slack]
                if _words_match(asr_norm, orig_norm):
                    orig_ptr += slack + 1
                    if sent_idx not in sent_starts:
                        sent_starts[sent_idx] = start_ms
                    sent_ends[sent_idx] = end_ms
                    matched = True
                    break

            if not matched and orig_ptr < len(orig_words):
                orig_ptr += 1  # Don't stall on unmatched ASR tokens.

        result: List[SentenceTimestamp] = []
        for sent_idx, sentence in enumerate(sentences):
            start_ms = sent_starts.get(sent_idx, 0)
            raw_end = sent_ends.get(sent_idx, 0)
            end_ms = raw_end + _END_BUFFER_MS if raw_end else 0
            result.append(
                SentenceTimestamp(
                    index=sent_idx,
                    text=sentence,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

        matched_count = sum(1 for ts in result if ts.start_ms > 0 or ts.end_ms > 0)
        logger.info(
            "Matched %d / %d sentences to audio timestamps.",
            matched_count,
            len(sentences),
        )
        return result

    @staticmethod
    def _interpolate_missing(
        timestamps: List[SentenceTimestamp],
    ) -> List[SentenceTimestamp]:
        """Fill in zero-timestamp sentences by linear interpolation.

        A sentence may have no matched words if the TTS skipped it or if the
        sequential matcher fell out of sync.  This method fills those gaps by
        distributing time uniformly within the interval bounded by the nearest
        non-zero neighbours.

        Args:
            timestamps: List of :class:`SentenceTimestamp` — modified in place.

        Returns:
            The same list with missing timestamps filled in.
        """
        n = len(timestamps)
        if n == 0:
            return timestamps

        # Find indices that have real timestamps (both start and end > 0).
        anchors = [i for i, ts in enumerate(timestamps) if ts.start_ms > 0 and ts.end_ms > 0]

        if not anchors:
            # No alignment at all — distribute uniformly assuming 3 words/sec.
            t = 0
            for ts in timestamps:
                word_count = len(ts.text.split())
                duration_ms = int(word_count / 3 * 1000)
                ts.start_ms = t
                ts.end_ms = t + duration_ms + _END_BUFFER_MS
                t = ts.end_ms
            return timestamps

        # Fill before first anchor.
        if anchors[0] > 0:
            gap_start = timestamps[anchors[0]].start_ms
            step = gap_start // (anchors[0] + 1) if anchors[0] > 0 else 0
            for i in range(anchors[0]):
                timestamps[i].start_ms = i * step
                timestamps[i].end_ms = (i + 1) * step

        # Fill between anchors.
        for a_idx in range(len(anchors) - 1):
            lo = anchors[a_idx]
            hi = anchors[a_idx + 1]
            if hi - lo <= 1:
                continue
            t_start = timestamps[lo].end_ms
            t_end = timestamps[hi].start_ms
            gap = hi - lo - 1
            step = (t_end - t_start) // (gap + 1) if gap > 0 else 0
            for offset, i in enumerate(range(lo + 1, hi), start=1):
                timestamps[i].start_ms = t_start + (offset - 1) * step
                timestamps[i].end_ms = t_start + offset * step + _END_BUFFER_MS

        # Fill after last anchor.
        if anchors[-1] < n - 1:
            t = timestamps[anchors[-1]].end_ms
            for i in range(anchors[-1] + 1, n):
                word_count = len(timestamps[i].text.split())
                duration_ms = int(word_count / 3 * 1000)
                timestamps[i].start_ms = t
                timestamps[i].end_ms = t + duration_ms + _END_BUFFER_MS
                t = timestamps[i].end_ms

        return timestamps
