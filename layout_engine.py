"""
Layout Engine — text wrapping, line map construction, and scroll target computation.

Design overview:
  - All sentences are pre-wrapped into a flat list of ``LineInfo`` objects that
    represent the "virtual canvas" — an infinitely tall column on which every
    line of the audiobook is placed at a fixed Y position.
  - Scroll targets are computed once at startup: for each sentence, the scroll
    position that places the sentence's first line at the *second* slot in the
    4-line visible window (so the reader sees one line of context above).
  - Text wrapping uses Pillow's ``textbbox`` for pixel-accurate measurement,
    never character-count estimation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Video / layout constants
# ---------------------------------------------------------------------------
VIDEO_WIDTH: int = 1920
VIDEO_HEIGHT: int = 1080
COLUMN_WIDTH: int = 1400                               # Pixel width of text column.
COLUMN_LEFT: int = (VIDEO_WIDTH - COLUMN_WIDTH) // 2  # 260 px left margin.
FONT_SIZE: int = 52
LINE_HEIGHT: int = 80
VISIBLE_LINES: int = 4
GRADIENT_HEIGHT: int = 80  # Fade zone above and below the 4-line window.

# Derived layout geometry.
TEXT_AREA_HEIGHT: int = VISIBLE_LINES * LINE_HEIGHT           # 320 px
TEXT_AREA_TOP: int = (VIDEO_HEIGHT - TEXT_AREA_HEIGHT) // 2  # 380 px
TEXT_AREA_BOTTOM: int = TEXT_AREA_TOP + TEXT_AREA_HEIGHT      # 700 px
RENDER_TOP: int = TEXT_AREA_TOP - GRADIENT_HEIGHT             # 300 px
RENDER_BOTTOM: int = TEXT_AREA_BOTTOM + GRADIENT_HEIGHT       # 780 px
RENDER_HEIGHT: int = RENDER_BOTTOM - RENDER_TOP               # 480 px

# Default font paths (populated by setup.sh).
_FONT_DIR = Path(__file__).parent / "fonts"
DEFAULT_FONT_REGULAR = str(_FONT_DIR / "NotoSerif-Regular.ttf")
DEFAULT_FONT_BOLD = str(_FONT_DIR / "NotoSerif-Bold.ttf")


@dataclass
class LineInfo:
    """A single rendered line on the virtual canvas.

    Attributes:
        text: The visible text string for this line.
        sentence_idx: Index of the parent sentence in the full sentence list.
        virtual_y: Y coordinate in the virtual canvas (0 = top of document).
        is_first_of_sentence: True if this is the opening line of the sentence.
    """

    text: str
    sentence_idx: int
    virtual_y: int
    is_first_of_sentence: bool = field(default=False)


class LayoutEngine:
    """Builds the full line map and scroll-target table for an audiobook.

    The line map is computed once and shared with the :class:`~frame_renderer.FrameRenderer`
    for every frame.

    Example::

        engine = LayoutEngine(sentences, font_path="fonts/NotoSerif-Regular.ttf")
        line_map = engine.line_map
        scroll_y = engine.get_scroll_target(sentence_idx=5)
    """

    def __init__(
        self,
        sentences: List[str],
        font_path: str = DEFAULT_FONT_REGULAR,
        font_size: int = FONT_SIZE,
        line_height: int = LINE_HEIGHT,
        column_width: int = COLUMN_WIDTH,
        video_width: int = VIDEO_WIDTH,
        video_height: int = VIDEO_HEIGHT,
        visible_lines: int = VISIBLE_LINES,
        gradient_height: int = GRADIENT_HEIGHT,
    ) -> None:
        """Initialise the layout engine and build all data structures.

        Args:
            sentences: Ordered list of processed sentence strings.
            font_path: Path to the ``.ttf`` font file.
            font_size: Font size in pixels.
            line_height: Vertical spacing between lines in pixels.
            column_width: Maximum text column width in pixels.
            video_width: Output video width in pixels.
            video_height: Output video height in pixels.
            visible_lines: Number of text lines shown simultaneously.
            gradient_height: Height of the fade zone above/below the text window.

        Raises:
            FileNotFoundError: If ``font_path`` does not exist.
        """
        if not Path(font_path).exists():
            raise FileNotFoundError(
                f"Font file not found: {font_path}\n"
                "Run video_setup.sh to download Noto Serif fonts."
            )

        self._sentences = sentences
        self._font_size = font_size
        self._line_height = line_height
        self._column_width = column_width
        self._visible_lines = visible_lines
        self._gradient_height = gradient_height

        # Derived geometry (may differ from module-level constants if custom args).
        self._text_area_height = visible_lines * line_height
        self._text_area_top = (video_height - self._text_area_height) // 2
        self._column_left = (video_width - column_width) // 2

        # Load font — used both for layout measurement and by FrameRenderer.
        self.font: ImageFont.FreeTypeFont = ImageFont.truetype(font_path, font_size)

        # Tiny measurement surface — created once, reused for all textbbox calls.
        self._measure_img = Image.new("RGB", (1, 1))
        self._measure_draw = ImageDraw.Draw(self._measure_img)

        # Build data structures.
        self.line_map: List[LineInfo] = self._build_line_map()
        self._scroll_targets: Dict[int, float] = self._build_scroll_targets()

        logger.info(
            "LayoutEngine: %d sentences → %d lines.  "
            "Virtual canvas height: %d px.",
            len(sentences),
            len(self.line_map),
            self.line_map[-1].virtual_y + line_height if self.line_map else 0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scroll_target(self, sentence_idx: int) -> float:
        """Return the scroll_y that places sentence ``sentence_idx`` at slot 1.

        Slot 1 is the second line from the top of the 4-line window, providing
        one line of context above the active sentence.

        Args:
            sentence_idx: Zero-based index of the sentence to scroll to.

        Returns:
            Virtual canvas Y coordinate for the top of the visible window,
            as a float (sub-pixel precision is preserved through interpolation).
        """
        return self._scroll_targets.get(sentence_idx, 0.0)

    @property
    def total_virtual_height(self) -> int:
        """Height of the virtual canvas in pixels.

        Returns:
            Y position of the bottom of the last line.
        """
        if not self.line_map:
            return 0
        return self.line_map[-1].virtual_y + self._line_height

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_line_map(self) -> List[LineInfo]:
        """Wrap all sentences and assign virtual-canvas Y positions.

        Returns:
            Ordered list of :class:`LineInfo` objects covering every sentence.
        """
        line_map: List[LineInfo] = []
        current_y = 0

        for sent_idx, sentence in enumerate(self._sentences):
            wrapped = self._wrap_sentence(sentence)
            for line_idx, line_text in enumerate(wrapped):
                line_map.append(
                    LineInfo(
                        text=line_text,
                        sentence_idx=sent_idx,
                        virtual_y=current_y,
                        is_first_of_sentence=(line_idx == 0),
                    )
                )
                current_y += self._line_height

        return line_map

    def _build_scroll_targets(self) -> Dict[int, float]:
        """Pre-compute scroll_y for every sentence.

        The scroll target for sentence N places N's first line at slot 1 of the
        visible window.  In pixel terms::

            scroll_y = sentence_first_line.virtual_y - line_height

        This ensures there is always one line of preceding context visible above
        the highlighted sentence.

        Returns:
            Dict mapping sentence_idx → scroll_y.
        """
        targets: Dict[int, float] = {}
        for line in self.line_map:
            if line.is_first_of_sentence and line.sentence_idx not in targets:
                # Place first line at slot 1 (offset = 1 × line_height from top).
                scroll_y = float(line.virtual_y - self._line_height)
                # Clamp: never scroll above the canvas top.
                targets[line.sentence_idx] = max(0.0, scroll_y)
        return targets

    def _wrap_sentence(self, text: str) -> List[str]:
        """Wrap a sentence into lines that fit within the column width.

        Uses Pillow's ``textbbox`` for pixel-accurate width measurement.
        Never splits mid-word.

        Args:
            text: Sentence text (already cleaned and normalised).

        Returns:
            List of line strings; at least one element.
        """
        words = text.split()
        if not words:
            return [""]

        lines: List[str] = []
        current_words: List[str] = []

        for word in words:
            candidate = " ".join(current_words + [word])
            bbox = self._measure_draw.textbbox((0, 0), candidate, font=self.font)
            width = bbox[2] - bbox[0]

            if width <= self._column_width:
                current_words.append(word)
            else:
                if current_words:
                    lines.append(" ".join(current_words))
                current_words = [word]

        if current_words:
            lines.append(" ".join(current_words))

        return lines
