"""
Frame Renderer — composites a single 1920×1080 BGR frame for a given scroll state.

Pipeline per frame:
  1. Fill PIL Image with background colour.
  2. Determine which virtual-canvas lines fall within the extended render zone
     (text area ± gradient height).
  3. Draw each visible line with PIL, colouring highlighted-sentence lines
     in the active colour and all other lines in the base text colour.
  4. Convert PIL Image to NumPy (RGB) array.
  5. Apply top and bottom gradient fades using vectorised NumPy blend.
  6. Return the frame in BGR channel order (OpenCV convention).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw  # type: ignore[import]

from layout_engine import (
    COLUMN_LEFT,
    GRADIENT_HEIGHT,
    LINE_HEIGHT,
    RENDER_BOTTOM,
    RENDER_TOP,
    TEXT_AREA_BOTTOM,
    TEXT_AREA_TOP,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    LayoutEngine,
    LineInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour defaults
# ---------------------------------------------------------------------------
_DEFAULT_BG: str = "#1a1a1a"
_DEFAULT_TEXT: str = "#e8e8e8"
_DEFAULT_HIGHLIGHT: str = "#ff4444"


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex colour string to an (R, G, B) integer tuple.

    Args:
        hex_color: Colour string like ``'#1a1a1a'`` or ``'1a1a1a'``.

    Returns:
        Tuple of (red, green, blue) in [0, 255].
    """
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class FrameRenderer:
    """Renders a single video frame given scroll position and active sentence.

    The renderer maintains a persistent PIL Image that is refilled each call
    rather than reallocated, saving repeated heap allocations over millions of
    frames.

    Example::

        renderer = FrameRenderer(layout_engine)
        bgr_frame = renderer.render(scroll_y=240.0, active_sentence_idx=3)
    """

    def __init__(
        self,
        layout_engine: LayoutEngine,
        bg_color: str = _DEFAULT_BG,
        text_color: str = _DEFAULT_TEXT,
        highlight_color: str = _DEFAULT_HIGHLIGHT,
        video_width: int = VIDEO_WIDTH,
        video_height: int = VIDEO_HEIGHT,
        gradient_height: int = GRADIENT_HEIGHT,
        text_area_top: int = TEXT_AREA_TOP,
        text_area_bottom: int = TEXT_AREA_BOTTOM,
        column_left: int = COLUMN_LEFT,
        line_height: int = LINE_HEIGHT,
    ) -> None:
        """Initialise the renderer.

        Args:
            layout_engine: Fully built :class:`~layout_engine.LayoutEngine` instance.
            bg_color: Background hex colour string.
            text_color: Inactive text hex colour string.
            highlight_color: Active-sentence hex colour string.
            video_width: Frame width in pixels.
            video_height: Frame height in pixels.
            gradient_height: Height of fade zones above and below the text area.
            text_area_top: Y coordinate of the top of the 4-line text area.
            text_area_bottom: Y coordinate of the bottom of the 4-line text area.
            column_left: Left edge X coordinate of the text column.
            line_height: Vertical line spacing in pixels.
        """
        self._layout = layout_engine
        self._line_map: List[LineInfo] = layout_engine.line_map
        self._font = layout_engine.font

        self._bg_rgb = _hex_to_rgb(bg_color)
        self._text_rgb = _hex_to_rgb(text_color)
        self._highlight_rgb = _hex_to_rgb(highlight_color)

        self._video_width = video_width
        self._video_height = video_height
        self._gradient_height = gradient_height
        self._text_area_top = text_area_top
        self._text_area_bottom = text_area_bottom
        self._column_left = column_left
        self._line_height = line_height

        # Extended render zone (text area ± gradient zone).
        self._render_top = text_area_top - gradient_height
        self._render_bottom = text_area_bottom + gradient_height

        # Pre-compute background colour as float32 array for gradient blending.
        self._bg_float = np.array(self._bg_rgb, dtype=np.float32)

        # Pre-build gradient alpha ramps (shape: (gradient_height, 1, 1)).
        # Top ramp: 0 → 1 (fade in from background to text as y increases).
        self._top_alpha = np.linspace(0.0, 1.0, gradient_height, dtype=np.float32)[
            :, np.newaxis, np.newaxis
        ]
        # Bottom ramp: 1 → 0 (fade out from text to background as y increases).
        self._bottom_alpha = np.linspace(1.0, 0.0, gradient_height, dtype=np.float32)[
            :, np.newaxis, np.newaxis
        ]

        # Persistent PIL Image (refilled each frame to avoid reallocation).
        self._img = Image.new("RGB", (video_width, video_height), self._bg_rgb)
        self._draw = ImageDraw.Draw(self._img)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        scroll_y: float,
        active_sentence_idx: int,
    ) -> np.ndarray:
        """Render a single frame and return it as a BGR NumPy array.

        Args:
            scroll_y: Current virtual-canvas Y coordinate at the top of the
                text-area window.  Sub-pixel values are rounded to the nearest
                integer for rendering.
            active_sentence_idx: Index of the sentence currently being spoken.
                All lines belonging to this sentence are rendered in the
                highlight colour.

        Returns:
            NumPy array of shape ``(video_height, video_width, 3)`` in BGR
            channel order, dtype ``uint8``.
        """
        scroll_int = round(scroll_y)

        # Refill background.
        self._draw.rectangle(
            [(0, 0), (self._video_width - 1, self._video_height - 1)],
            fill=self._bg_rgb,
        )

        # Determine visible virtual-canvas range (extended by gradient zones).
        visible_virtual_min = scroll_int - self._gradient_height
        visible_virtual_max = scroll_int + (self._text_area_bottom - self._text_area_top) + self._gradient_height

        for line in self._line_map:
            vy = line.virtual_y
            if vy < visible_virtual_min or vy >= visible_virtual_max:
                continue

            # Map virtual Y to frame Y.
            frame_y = self._text_area_top + (vy - scroll_int)

            # Only draw if within the full extended render zone.
            if frame_y < self._render_top or frame_y >= self._render_bottom:
                continue

            colour = (
                self._highlight_rgb
                if line.sentence_idx == active_sentence_idx
                else self._text_rgb
            )

            self._draw.text(
                (self._column_left, frame_y),
                line.text,
                font=self._font,
                fill=colour,
            )

        # Convert PIL → NumPy RGB.
        frame_rgb = np.array(self._img, dtype=np.float32)

        # Apply gradient fades (vectorised NumPy blend — no pixel loops).
        self._apply_gradient(frame_rgb)

        # Convert RGB → BGR for OpenCV and return as uint8.
        return frame_rgb[:, :, ::-1].astype(np.uint8)

    def render_at_virtual_y(
        self,
        scroll_y: float,
        active_sentence_idx: int,
    ) -> np.ndarray:
        """Alias for :meth:`render` — mirrors the video_assembler call interface.

        Args:
            scroll_y: See :meth:`render`.
            active_sentence_idx: See :meth:`render`.

        Returns:
            BGR NumPy array — see :meth:`render`.
        """
        return self.render(scroll_y, active_sentence_idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_gradient(self, frame: np.ndarray) -> None:
        """Apply top and bottom gradient fades in-place.

        Blends the top ``gradient_height`` rows from fully background-coloured
        (alpha=0) to fully text-visible (alpha=1), and the bottom rows in
        reverse — all done with vectorised NumPy operations.

        Args:
            frame: Float32 NumPy array of shape (H, W, 3) in RGB order.
                Modified in-place.
        """
        # Top gradient zone.
        top_start = self._render_top
        top_end = min(self._text_area_top, self._video_height)
        if top_start >= 0 and top_end <= self._video_height and top_start < top_end:
            zone = frame[top_start:top_end]
            alpha = self._top_alpha[: top_end - top_start]
            frame[top_start:top_end] = zone * alpha + self._bg_float * (1.0 - alpha)

        # Bottom gradient zone.
        bot_start = max(self._text_area_bottom, 0)
        bot_end = min(self._render_bottom, self._video_height)
        if bot_start >= 0 and bot_end <= self._video_height and bot_start < bot_end:
            zone = frame[bot_start:bot_end]
            alpha = self._bottom_alpha[: bot_end - bot_start]
            frame[bot_start:bot_end] = zone * alpha + self._bg_float * (1.0 - alpha)
