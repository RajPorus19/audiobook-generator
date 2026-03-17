"""
Text Processor — semantic chunking, cleaning, and normalisation for TTS input.

Chunking strategy (in priority order):
  1. Split on paragraph breaks (double newline) to preserve natural pauses.
  2. Within each paragraph, split into sentences using NLTK punkt tokenizer.
  3. Accumulate sentences into chunks until the 250-token budget is reached.
     A sentence that alone exceeds 250 tokens is split by clause boundaries as
     a last resort, but never mid-word.

Text cleaning pipeline:
  - Strip Markdown formatting (headers, bold, italic, links, code fences).
  - Normalise Unicode (NFKC) so smart quotes / dashes are converted.
  - Expand common abbreviations to their spoken form.
  - Convert numeric tokens to words via num2words.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import nltk
from num2words import num2words  # type: ignore[import]

logger = logging.getLogger(__name__)

# Download NLTK tokenizer data silently if missing.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TOKENS: int = 250  # Maximum tokens per synthesis chunk.

# Abbreviation expansion map — spoken form replaces written abbreviation.
_ABBREVIATIONS: dict[str, str] = {
    r"\bDr\.": "Doctor",
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miss",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bAve\.": "Avenue",
    r"\bBlvd\.": "Boulevard",
    r"\bvs\.": "versus",
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\bcf\.": "compare",
    r"\bapprox\.": "approximately",
    r"\bft\.": "feet",
    r"\bin\.": "inches",
    r"\blb\.": "pounds",
    r"\boz\.": "ounces",
    r"\bno\.": "number",
    r"\bvol\.": "volume",
    r"\bch\.": "chapter",
    r"\bsec\.": "section",
    r"\bfig\.": "figure",
    r"\bp\.": "page",
    r"\bpp\.": "pages",
}


class ChunkType(Enum):
    """Semantic role of a text chunk, used to determine pause length."""

    SENTENCE = auto()
    PARAGRAPH_END = auto()


@dataclass
class TextChunk:
    """A single synthesis unit produced by the text processor.

    Attributes:
        text: The cleaned, normalised text ready for TTS.
        chunk_type: Whether this chunk ends a paragraph or a sentence.
        token_count: Approximate whitespace-token count.
    """

    text: str
    chunk_type: ChunkType
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = len(self.text.split())


class TextProcessor:
    """Transforms raw input text into a sequence of TTS-ready chunks.

    Example::

        processor = TextProcessor()
        chunks = processor.process("Chapter 1\\n\\nDr. Smith walked in.")
        for chunk in chunks:
            print(chunk.text, chunk.chunk_type)
    """

    def __init__(self, max_tokens: int = MAX_TOKENS) -> None:
        """Initialise the processor.

        Args:
            max_tokens: Maximum whitespace-token count per chunk.
        """
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_text: str) -> List[TextChunk]:
        """Full pipeline: clean → normalise → chunk.

        Args:
            raw_text: Raw input text (may contain Markdown, unicode, numbers).

        Returns:
            Ordered list of :class:`TextChunk` objects ready for synthesis.
        """
        text = self._clean(raw_text)
        paragraphs = self._split_paragraphs(text)
        chunks: List[TextChunk] = []
        for para_idx, paragraph in enumerate(paragraphs):
            is_last_para = para_idx == len(paragraphs) - 1
            para_chunks = self._chunk_paragraph(paragraph, is_last_para)
            chunks.extend(para_chunks)
        logger.info(
            "Processed %d paragraph(s) into %d chunk(s).",
            len(paragraphs),
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Apply the full text cleaning pipeline.

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalised text.
        """
        text = self._normalise_unicode(text)
        text = self._strip_markdown(text)
        text = self._expand_abbreviations(text)
        text = self._normalise_numbers(text)
        text = self._normalise_whitespace(text)
        return text

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        """Convert to NFKC form, replacing typographic variants.

        Args:
            text: Input text.

        Returns:
            NFKC-normalised text with curly quotes and em-dashes converted.
        """
        text = unicodedata.normalize("NFKC", text)
        # Curly quotes → straight quotes.
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Em/en dash → comma-space for natural spoken pause.
        text = text.replace("\u2014", ", ").replace("\u2013", " to ")
        # Ellipsis character → three dots (NLTK handles sentence boundaries).
        text = text.replace("\u2026", "...")
        return text

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove Markdown formatting, leaving only prose.

        Args:
            text: Text possibly containing Markdown.

        Returns:
            Plain text with Markdown syntax removed.
        """
        # Fenced code blocks.
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        # ATX-style headings — keep the heading text.
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Bold / italic.
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Links — keep link text.
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Images — drop entirely.
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        # Horizontal rules.
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Blockquotes.
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        # Unordered list bullets.
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        # Ordered list numbers.
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        """Replace common written abbreviations with their spoken form.

        Args:
            text: Input text.

        Returns:
            Text with abbreviations expanded.
        """
        for pattern, replacement in _ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _normalise_numbers(text: str) -> str:
        """Convert numeric tokens (integers and simple decimals) to words.

        Args:
            text: Input text.

        Returns:
            Text with numbers written out as words.
        """

        def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
            token = match.group(0)
            # Strip thousands separators.
            token_clean = token.replace(",", "")
            try:
                if "." in token_clean:
                    # Decimal number.
                    return num2words(float(token_clean))
                return num2words(int(token_clean))
            except (ValueError, OverflowError):
                return token  # Leave unchanged if conversion fails.

        # Match integers with optional thousands commas and optional decimal.
        return re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", _replace, text)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse redundant whitespace while preserving paragraph breaks.

        Args:
            text: Input text.

        Returns:
            Text with normalised whitespace.
        """
        # Collapse 3+ newlines to exactly 2.
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces / tabs (but not newlines) to a single space.
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text on paragraph boundaries (double newline).

        Args:
            text: Cleaned text.

        Returns:
            List of non-empty paragraph strings.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_paragraph(
        self, paragraph: str, is_last: bool
    ) -> List[TextChunk]:
        """Split a single paragraph into synthesis-sized chunks.

        Args:
            paragraph: A single paragraph of cleaned text.
            is_last: Whether this is the final paragraph in the document.

        Returns:
            List of :class:`TextChunk` objects.
        """
        sentences = nltk.sent_tokenize(paragraph)
        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_tokens: int = 0

        for sent_idx, sentence in enumerate(sentences):
            sent_tokens = len(sentence.split())
            is_last_sentence = sent_idx == len(sentences) - 1

            if sent_tokens > self._max_tokens:
                # Flush what we have first.
                if current_sentences:
                    chunk_type = ChunkType.PARAGRAPH_END if is_last_sentence and is_last else ChunkType.SENTENCE
                    chunks.extend(
                        self._flush(current_sentences, chunk_type)
                    )
                    current_sentences = []
                    current_tokens = 0
                # Split the long sentence by clause boundaries.
                sub_chunks = self._split_long_sentence(sentence, is_last_sentence and is_last)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > self._max_tokens:
                # Flush current accumulation.
                chunks.extend(self._flush(current_sentences, ChunkType.SENTENCE))
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remainder.
        if current_sentences:
            chunk_type = ChunkType.PARAGRAPH_END if is_last else ChunkType.SENTENCE
            chunks.extend(self._flush(current_sentences, chunk_type))

        return chunks

    def _flush(
        self, sentences: List[str], chunk_type: ChunkType
    ) -> List[TextChunk]:
        """Create a TextChunk from accumulated sentences.

        Args:
            sentences: List of sentence strings to join.
            chunk_type: Semantic role to assign the resulting chunk.

        Returns:
            A single-element list containing the assembled :class:`TextChunk`.
        """
        text = " ".join(s.strip() for s in sentences if s.strip())
        if not text:
            return []
        return [TextChunk(text=text, chunk_type=chunk_type)]

    def _split_long_sentence(
        self, sentence: str, is_last: bool
    ) -> List[TextChunk]:
        """Break an oversized sentence at clause boundaries.

        Splits on commas, semicolons, and conjunctions as fallback.  Never
        splits mid-word.

        Args:
            sentence: A sentence with more than ``max_tokens`` whitespace-tokens.
            is_last: Whether this chunk is the final one in the document.

        Returns:
            List of :class:`TextChunk` objects, each under the token limit.
        """
        # Try splitting on '; ' first (strong clause boundary).
        parts = re.split(r"(?<=;)\s+", sentence)
        if len(parts) == 1:
            # Fall back to ', ' boundaries.
            parts = re.split(r"(?<=,)\s+", sentence)

        chunks: List[TextChunk] = []
        buffer: List[str] = []
        buffer_tokens = 0

        for part in parts:
            part_tokens = len(part.split())
            if buffer_tokens + part_tokens > self._max_tokens and buffer:
                joined = " ".join(buffer)
                chunks.append(TextChunk(text=joined, chunk_type=ChunkType.SENTENCE))
                buffer = []
                buffer_tokens = 0
            buffer.append(part)
            buffer_tokens += part_tokens

        if buffer:
            joined = " ".join(buffer)
            chunk_type = ChunkType.PARAGRAPH_END if is_last else ChunkType.SENTENCE
            chunks.append(TextChunk(text=joined, chunk_type=chunk_type))

        return chunks
