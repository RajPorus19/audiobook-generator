"""
validate_sync.py — Verify that sentence highlights are correctly timed in the output video.

Method:
  1. Load alignment cache and layout data.
  2. Simulate the scroll state at 20 randomly sampled sentence activation times.
  3. For each sample, extract the corresponding video frame.
  4. Check that the expected line Y positions contain pixels in the highlight colour.
  5. Print a sync accuracy report.

Usage:
    python validate_sync.py --audio narration.mp3 --text text.txt --video video.mp4
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np

from aligner import Aligner, SentenceTimestamp
from layout_engine import LayoutEngine, LineInfo, TEXT_AREA_TOP, LINE_HEIGHT, COLUMN_LEFT
from video_assembler import FPS, MS_PER_FRAME, _ScrollState, _find_active_sentence, ease_in_out_cubic
from frame_renderer import _hex_to_rgb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Colour-matching tolerance (Euclidean distance in RGB space).
# ---------------------------------------------------------------------------
_COLOUR_TOLERANCE: int = 30  # pixels within 30 units of the target colour pass.


def _pixel_matches(pixel_bgr: np.ndarray, target_rgb: Tuple[int, int, int]) -> bool:
    """Check whether a BGR pixel is close to the target RGB colour.

    Args:
        pixel_bgr: Array of shape (3,) or (1, 3) in BGR order.
        target_rgb: Target colour as (R, G, B) integers.

    Returns:
        True if the Euclidean distance is within ``_COLOUR_TOLERANCE``.
    """
    b, g, r = int(pixel_bgr[0]), int(pixel_bgr[1]), int(pixel_bgr[2])
    tr, tg, tb = target_rgb
    dist = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
    return dist <= _COLOUR_TOLERANCE


def _simulate_scroll_at(
    target_ms: float,
    timestamps: List[SentenceTimestamp],
    layout_engine: LayoutEngine,
    scroll_speed_ms: float = 300.0,
) -> Tuple[float, int]:
    """Replay the scroll state by simulating every frame up to ``target_ms``.

    This replicates the exact logic in ``VideoAssembler._write_frames`` so the
    predicted scroll position matches what was actually rendered.

    Args:
        target_ms: Audio time in milliseconds to evaluate.
        timestamps: Sentence timing data.
        layout_engine: Provides scroll targets.
        scroll_speed_ms: Scroll transition duration used during rendering.

    Returns:
        Tuple of (scroll_y, active_sentence_idx) at ``target_ms``.
    """
    scroll = _ScrollState(
        initial_y=layout_engine.get_scroll_target(0),
        transition_ms=scroll_speed_ms,
    )
    prev_active = -1
    frame_count = int(target_ms / MS_PER_FRAME) + 1

    for frame_idx in range(frame_count):
        current_ms = frame_idx * MS_PER_FRAME
        active_idx = _find_active_sentence(timestamps, current_ms)
        if active_idx != prev_active:
            new_target = layout_engine.get_scroll_target(active_idx)
            scroll.start_transition(new_target, current_ms)
            prev_active = active_idx
        scroll.update(current_ms)

    return scroll.current_y, prev_active


def _lines_for_sentence(
    sentence_idx: int,
    line_map: List[LineInfo],
) -> List[LineInfo]:
    """Return all lines belonging to the given sentence.

    Args:
        sentence_idx: Zero-based index of the target sentence.
        line_map: Full line map from the layout engine.

    Returns:
        Ordered list of :class:`~layout_engine.LineInfo` objects for the sentence.
    """
    return [l for l in line_map if l.sentence_idx == sentence_idx]


class SyncValidator:
    """Validates synchronisation accuracy between the alignment data and the video.

    Example::

        validator = SyncValidator(
            video_path="video.mp4",
            timestamps=timestamps,
            layout_engine=layout_engine,
        )
        validator.run(n_samples=20)
    """

    def __init__(
        self,
        video_path: str,
        timestamps: List[SentenceTimestamp],
        layout_engine: LayoutEngine,
        highlight_color: str = "#ff4444",
        scroll_speed_ms: float = 300.0,
        n_samples: int = 20,
        seed: int = 0,
    ) -> None:
        """Initialise the validator.

        Args:
            video_path: Path to the rendered MP4.
            timestamps: Sentence timing data.
            layout_engine: Must match what was used during rendering.
            highlight_color: Hex colour expected for the active sentence.
            scroll_speed_ms: Scroll speed used during rendering.
            n_samples: Number of sentences to sample.
            seed: Random seed for reproducibility.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._video_path = video_path
        self._timestamps = timestamps
        self._layout = layout_engine
        self._highlight_rgb = _hex_to_rgb(highlight_color)
        self._scroll_speed_ms = scroll_speed_ms
        self._n_samples = n_samples
        self._seed = seed

        # Open video for random-access frame extraction.
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or FPS

    def run(self) -> Dict:
        """Run the full validation and print a report.

        Returns:
            Dict with keys ``pass_count``, ``fail_count``, ``frame_offsets``,
            ``mean_offset_ms``, ``max_offset_ms``.
        """
        random.seed(self._seed)
        # Sample sentences that have real timestamps (skip the very first/last).
        eligible = [
            ts for ts in self._timestamps
            if ts.start_ms > 0 and ts.end_ms > ts.start_ms + 100
        ]
        if len(eligible) == 0:
            logger.warning("No eligible sentences for validation.")
            return {}

        samples = random.sample(eligible, min(self._n_samples, len(eligible)))
        samples.sort(key=lambda s: s.start_ms)

        results = []
        print("\n" + "─" * 72)
        print(f"  Sync Validation — {len(samples)} random samples")
        print("─" * 72)
        print(f"  {'Sent':>5}  {'Time':>8}  {'ScrollY':>8}  {'Lines':>6}  {'Pass':>5}  Reason")
        print("─" * 72)

        for ts in samples:
            result = self._validate_sentence(ts)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            print(
                f"  {ts.index:>5}  {ts.start_ms / 1000:>7.2f}s  "
                f"{result['scroll_y']:>8.1f}  "
                f"{result['line_count']:>6}  "
                f"{status:>5}  {result.get('reason', '')}"
            )

        pass_count = sum(1 for r in results if r["pass"])
        fail_count = len(results) - pass_count
        frame_offsets = [r.get("frame_offset_ms", 0) for r in results if "frame_offset_ms" in r]
        mean_offset = sum(frame_offsets) / len(frame_offsets) if frame_offsets else 0.0
        max_offset = max(frame_offsets) if frame_offsets else 0.0

        print("─" * 72)
        print(f"  Passed  : {pass_count} / {len(results)}")
        print(f"  Failed  : {fail_count}")
        print(f"  Mean frame offset : {mean_offset:.1f} ms")
        print(f"  Max  frame offset : {max_offset:.1f} ms")
        print("─" * 72 + "\n")

        self._cap.release()

        return {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "frame_offsets": frame_offsets,
            "mean_offset_ms": mean_offset,
            "max_offset_ms": max_offset,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_sentence(self, ts: SentenceTimestamp) -> Dict:
        """Validate a single sentence activation point.

        Extracts the video frame at ``ts.start_ms``, predicts which lines
        should be highlighted, and checks those lines for the highlight colour.

        Args:
            ts: Sentence timestamp to validate.

        Returns:
            Dict with validation result fields.
        """
        check_ms = ts.start_ms + 50  # Check 50ms after activation.
        scroll_y, active_idx = _simulate_scroll_at(
            check_ms,
            self._timestamps,
            self._layout,
            self._scroll_speed_ms,
        )

        # Extract video frame.
        frame_number = int(check_ms / 1000.0 * self._video_fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return {
                "pass": False,
                "reason": "Could not read frame",
                "scroll_y": scroll_y,
                "line_count": 0,
            }

        lines = _lines_for_sentence(ts.index, self._layout.line_map)
        if not lines:
            return {
                "pass": False,
                "reason": "No lines in line map",
                "scroll_y": scroll_y,
                "line_count": 0,
            }

        # Check each visible line for highlight colour.
        scroll_int = round(scroll_y)
        hits = 0
        for line in lines:
            frame_y = TEXT_AREA_TOP + (line.virtual_y - scroll_int)
            if frame_y < 0 or frame_y >= frame.shape[0]:
                continue  # Line is off-screen.
            # Sample a cluster of pixels in the middle of the line.
            sample_x = COLUMN_LEFT + 50
            if sample_x >= frame.shape[1]:
                continue
            pixel = frame[frame_y + LINE_HEIGHT // 3, sample_x]
            if _pixel_matches(pixel, self._highlight_rgb):
                hits += 1

        passed = hits > 0
        reason = f"{hits}/{len(lines)} lines highlighted" if passed else "no highlight found"

        return {
            "pass": passed,
            "reason": reason,
            "scroll_y": scroll_y,
            "line_count": len(lines),
            "frame_offset_ms": abs(check_ms - ts.start_ms),
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="validate-sync",
        description="Validate sentence-to-video synchronisation accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, metavar="PATH", help="Narration audio file.")
    parser.add_argument("--text", required=True, metavar="PATH", help="Source text file.")
    parser.add_argument("--video", required=True, metavar="PATH", help="Output video to validate.")
    parser.add_argument(
        "--font",
        default=str(Path(__file__).parent / "fonts" / "NotoSerif-Regular.ttf"),
        metavar="PATH",
        help="Font file (must match the one used for rendering).",
    )
    parser.add_argument("--samples", default=20, type=int, help="Number of sentences to sample.")
    parser.add_argument(
        "--highlight_color",
        default="#ff4444",
        metavar="HEX",
        help="Highlight colour used during rendering.",
    )
    parser.add_argument(
        "--scroll_speed_ms",
        default=300,
        type=int,
        help="Scroll speed used during rendering.",
    )
    return parser


def main() -> None:
    """CLI entry point for the sync validator."""
    args = _build_arg_parser().parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    key = Aligner.cache_key(args.audio, args.text)
    cache_file = str(Path(args.audio).with_suffix(f".align_{key[:16]}.json"))
    timestamps = Aligner.load_cache(cache_file, key)
    if timestamps is None:
        logger.error(
            "Alignment cache not found: %s\n"
            "Run video_main.py first to generate the cache.",
            cache_file,
        )
        return

    sentences = [ts.text for ts in timestamps]
    layout_engine = LayoutEngine(sentences=sentences, font_path=args.font)

    validator = SyncValidator(
        video_path=args.video,
        timestamps=timestamps,
        layout_engine=layout_engine,
        highlight_color=args.highlight_color,
        scroll_speed_ms=float(args.scroll_speed_ms),
        n_samples=args.samples,
    )
    validator.run()


if __name__ == "__main__":
    main()
