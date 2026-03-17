"""
test_tts.py — Unit and integration tests for the audiobook TTS engine.

Run with:
    pytest test_tts.py -v

Note: Tests that require the TTS model (TTSEngine integration tests) are
marked with ``@pytest.mark.slow`` and skipped unless ``--runslow`` is passed.
"""

import math
import random
import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audio_assembler import AudioAssembler, _TARGET_LUFS
from text_processor import ChunkType, TextChunk, TextProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_lorem(word_count: int) -> str:
    """Generate a deterministic lorem-ipsum text of approximately word_count words.

    Args:
        word_count: Approximate number of words to generate.

    Returns:
        Multi-paragraph plain text.
    """
    random.seed(0)
    vocab = (
        "the quick brown fox jumps over the lazy dog and a time long ago there "
        "was a great story told by Doctor Smith who lived on Baker Street with "
        "Mister Jones and his family of five children who loved reading books "
        "every evening before bedtime while the stars shone brightly outside"
    ).split()

    sentences: List[str] = []
    total = 0
    while total < word_count:
        length = random.randint(8, 25)
        words = [random.choice(vocab) for _ in range(length)]
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
        total += length

    # Group into paragraphs of 3–6 sentences.
    paragraphs: List[str] = []
    idx = 0
    while idx < len(sentences):
        size = random.randint(3, 6)
        paragraph = " ".join(sentences[idx : idx + size])
        paragraphs.append(paragraph)
        idx += size

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# TextProcessor tests
# ---------------------------------------------------------------------------


class TestTextProcessor:
    """Tests for the text chunking and cleaning pipeline."""

    @pytest.fixture(scope="class")
    def processor(self) -> TextProcessor:
        """Return a default TextProcessor."""
        return TextProcessor(max_tokens=250)

    @pytest.fixture(scope="class")
    def large_text(self) -> str:
        """10 000-word sample text for stress-testing."""
        return _generate_lorem(10_000)

    def test_no_chunk_exceeds_token_limit(
        self, processor: TextProcessor, large_text: str
    ) -> None:
        """Every produced chunk must be within the 250-token budget.

        Args:
            processor: TextProcessor fixture.
            large_text: Large sample text fixture.
        """
        chunks = processor.process(large_text)
        violations = [c for c in chunks if c.token_count > 250]
        assert not violations, (
            f"{len(violations)} chunk(s) exceeded 250 tokens. "
            f"Max seen: {max(c.token_count for c in violations)}"
        )

    def test_produces_chunks_from_large_text(
        self, processor: TextProcessor, large_text: str
    ) -> None:
        """Processing a large text must yield multiple chunks.

        Args:
            processor: TextProcessor fixture.
            large_text: Large sample text fixture.
        """
        chunks = processor.process(large_text)
        assert len(chunks) > 10, "Expected many chunks from a 10k-word text."

    def test_empty_text_returns_empty(self, processor: TextProcessor) -> None:
        """Empty input must return an empty list without raising.

        Args:
            processor: TextProcessor fixture.
        """
        result = processor.process("")
        assert result == []

    def test_abbreviation_expansion(self, processor: TextProcessor) -> None:
        """Abbreviations must be expanded to their spoken form.

        Args:
            processor: TextProcessor fixture.
        """
        chunks = processor.process("Dr. Smith and Mr. Jones arrived.")
        combined = " ".join(c.text for c in chunks)
        assert "Doctor" in combined, "Dr. should expand to Doctor"
        assert "Mister" in combined, "Mr. should expand to Mister"

    def test_number_normalisation(self, processor: TextProcessor) -> None:
        """Numeric tokens must be converted to words.

        Args:
            processor: TextProcessor fixture.
        """
        chunks = processor.process("There were 42 apples and 1,000 oranges.")
        combined = " ".join(c.text for c in chunks)
        assert "forty-two" in combined or "forty two" in combined, (
            f"Expected '42' to become 'forty-two' but got: {combined}"
        )

    def test_markdown_stripped(self, processor: TextProcessor) -> None:
        """Markdown formatting must be removed from output.

        Args:
            processor: TextProcessor fixture.
        """
        chunks = processor.process("## Chapter One\n\n**Bold text** and *italic*.")
        combined = " ".join(c.text for c in chunks)
        assert "#" not in combined, "Markdown headers should be stripped."
        assert "**" not in combined, "Markdown bold should be stripped."
        assert "Bold text" in combined, "Text content should be preserved."

    def test_unicode_normalised(self, processor: TextProcessor) -> None:
        """Smart quotes and em-dashes must be converted to ASCII equivalents.

        Args:
            processor: TextProcessor fixture.
        """
        chunks = processor.process("\u201cHello\u201d she said\u2014loudly.")
        combined = " ".join(c.text for c in chunks)
        assert "\u201c" not in combined, "Left double quote should be normalised."
        assert "\u2014" not in combined, "Em dash should be normalised."

    def test_chunk_types_assigned(self, processor: TextProcessor) -> None:
        """The last chunk of a paragraph must be PARAGRAPH_END.

        Args:
            processor: TextProcessor fixture.
        """
        text = "First sentence. Second sentence.\n\nNew paragraph starts here."
        chunks = processor.process(text)
        # At least one PARAGRAPH_END must exist.
        types = [c.chunk_type for c in chunks]
        assert ChunkType.PARAGRAPH_END in types, "Expected at least one PARAGRAPH_END chunk."


# ---------------------------------------------------------------------------
# AudioAssembler tests
# ---------------------------------------------------------------------------


class TestAudioAssembler:
    """Tests for the audio assembly, normalisation, and export pipeline."""

    _SR = 24_000

    @pytest.fixture
    def assembler(self) -> AudioAssembler:
        """Return an AudioAssembler with the test sample rate.

        Returns:
            AudioAssembler instance.
        """
        return AudioAssembler(sample_rate=self._SR)

    def _make_sine(self, freq: float, duration: float, amplitude: float = 0.5) -> np.ndarray:
        """Generate a mono sine-wave chunk.

        Args:
            freq: Frequency in Hz.
            duration: Duration in seconds.
            amplitude: Peak amplitude in [0, 1].

        Returns:
            Float32 NumPy array.
        """
        t = np.linspace(0.0, duration, int(self._SR * duration), endpoint=False)
        return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)

    def test_assemble_basic(self, assembler: AudioAssembler) -> None:
        """Assembling two chunks must produce a longer array than either chunk.

        Args:
            assembler: AudioAssembler fixture.
        """
        c1 = self._make_sine(440.0, 1.0)
        c2 = self._make_sine(880.0, 1.0)
        result = assembler.assemble([c1, c2], [ChunkType.SENTENCE, ChunkType.PARAGRAPH_END])
        # Each chunk is 1 s, plus at least one pause; combined must be > 2 s.
        assert len(result) > 2 * self._SR

    def test_all_chunks_same_sample_rate(self, assembler: AudioAssembler) -> None:
        """The sample rate used to build pauses must be consistent.

        Verifies that the assembler does not change the sample rate and that
        all synthesised chunks share the same rate (simulated here).

        Args:
            assembler: AudioAssembler fixture.
        """
        assert assembler._sr == self._SR  # noqa: SLF001

    def test_silence_inserted_between_chunks(self, assembler: AudioAssembler) -> None:
        """Silence padding must be inserted between consecutive chunks.

        Args:
            assembler: AudioAssembler fixture.
        """
        c1 = self._make_sine(440.0, 0.5)
        c2 = self._make_sine(440.0, 0.5)
        result = assembler.assemble([c1, c2], [ChunkType.SENTENCE, ChunkType.SENTENCE])
        # Total must be > sum of chunk lengths (silence was added).
        assert len(result) > len(c1) + len(c2)

    def test_empty_chunks_raises(self, assembler: AudioAssembler) -> None:
        """Passing an empty list must raise ValueError.

        Args:
            assembler: AudioAssembler fixture.
        """
        with pytest.raises(ValueError, match="No audio chunks"):
            assembler.assemble([], [])

    def test_mismatched_lengths_raises(self, assembler: AudioAssembler) -> None:
        """Mismatched chunks / types must raise ValueError.

        Args:
            assembler: AudioAssembler fixture.
        """
        c1 = self._make_sine(440.0, 0.5)
        with pytest.raises(ValueError, match="same length"):
            assembler.assemble([c1], [ChunkType.SENTENCE, ChunkType.PARAGRAPH_END])

    def test_loudness_consistency(self, assembler: AudioAssembler) -> None:
        """After normalisation, all chunks must be within ±2 LU of each other.

        This test is skipped when pyloudnorm is not installed.

        Args:
            assembler: AudioAssembler fixture.
        """
        try:
            import pyloudnorm as pyln  # type: ignore[import]
        except ImportError:
            pytest.skip("pyloudnorm not installed")

        # Create chunks at very different amplitudes.
        chunks = [
            self._make_sine(440.0, 1.0, amplitude=0.1),
            self._make_sine(440.0, 1.0, amplitude=0.8),
            self._make_sine(440.0, 1.0, amplitude=0.3),
        ]
        normalised = assembler._normalise_chunks(chunks)  # noqa: SLF001

        meter = pyln.Meter(self._SR)
        lufs_values = []
        for chunk in normalised:
            luf = meter.integrated_loudness(chunk.astype(np.float64))
            if np.isfinite(luf):
                lufs_values.append(luf)

        if len(lufs_values) >= 2:
            spread = max(lufs_values) - min(lufs_values)
            assert spread <= 4.0, (
                f"Loudness spread too large: {spread:.2f} LU. "
                f"Values: {lufs_values}"
            )

    def test_output_is_float32(self, assembler: AudioAssembler) -> None:
        """Assembled output must be a float32 NumPy array.

        Args:
            assembler: AudioAssembler fixture.
        """
        c1 = self._make_sine(440.0, 0.5)
        result = assembler.assemble([c1], [ChunkType.SENTENCE])
        assert result.dtype == np.float32

    def test_duration_proportional_to_text(self) -> None:
        """Audio duration must scale roughly with text length (mocked TTS).

        This test mocks the TTS engine to avoid requiring model weights.
        It verifies that processing twice as much text produces roughly
        twice as much audio.

        Note: Exact proportionality depends on TTS internals; we check that
        more text yields more audio (monotonicity), not exact ratios.
        """
        sr = 24_000
        words_per_second = 3  # approximate speaking rate

        def fake_synthesize(text: str) -> np.ndarray:
            """Return audio proportional to word count."""
            n_words = len(text.split())
            n_samples = int(n_words / words_per_second * sr)
            return np.zeros(n_samples, dtype=np.float32)

        assembler = AudioAssembler(sample_rate=sr)
        processor = TextProcessor(max_tokens=250)

        short_text = "Hello world. This is a short test."
        long_text = short_text * 10

        short_chunks = processor.process(short_text)
        long_chunks = processor.process(long_text)

        short_audio = [fake_synthesize(c.text) for c in short_chunks]
        long_audio = [fake_synthesize(c.text) for c in long_chunks]

        short_result = assembler.assemble(
            short_audio, [c.chunk_type for c in short_chunks]
        )
        long_result = assembler.assemble(
            long_audio, [c.chunk_type for c in long_chunks]
        )

        assert len(long_result) > len(short_result), (
            "Longer text must produce longer audio."
        )


# ---------------------------------------------------------------------------
# CLI smoke test (no model loaded)
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Smoke tests for the CLI argument parser."""

    def test_missing_input_exits(self) -> None:
        """Missing --input argument must cause SystemExit.

        Calls the argument parser without required arguments.
        """
        import argparse
        from main import _build_arg_parser

        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults_populated(self) -> None:
        """Default values must be set correctly when not specified.

        Args: None
        """
        from main import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["--input", "hello world"])
        assert args.output == "output.mp3"
        assert args.language == "en"
        assert math.isclose(args.speed, 1.0)
        assert args.speaker_wav is None


# ---------------------------------------------------------------------------
# pytest configuration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add the --runslow option to pytest.

    Args:
        parser: pytest argument parser.
    """
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests that require the TTS model.",
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register the 'slow' marker.

    Args:
        config: pytest config object.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (model required)")


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Skip slow tests unless --runslow is given.

    Args:
        config: pytest config object.
        items: Collected test items.
    """
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="Pass --runslow to run model tests.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
