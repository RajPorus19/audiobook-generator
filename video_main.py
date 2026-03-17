"""
video_main.py — CLI entry point for the audio-synchronised scrolling-text video generator.

Usage examples:
    # Basic: generate video from existing narration
    python video_main.py --audio narration.mp3 --text text.txt --output video.mp4

    # Custom highlight colour and scroll speed
    python video_main.py --audio narration.mp3 --text text.txt \\
        --highlight_color "#ffcc00" --scroll_speed_ms 200

    # Skip alignment (cached) and use a custom font
    python video_main.py --audio narration.mp3 --text text.txt \\
        --font fonts/MyFont.ttf --output video.mp4
"""

import argparse
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from aligner import Aligner, SentenceTimestamp
from frame_renderer import FrameRenderer
from layout_engine import DEFAULT_FONT_REGULAR, LayoutEngine
from video_assembler import VideoAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _check_ffmpeg() -> None:
    """Verify that FFmpeg is installed and available on PATH.

    Raises:
        SystemExit: With an explanatory message if FFmpeg is missing.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ERROR: ffmpeg binary not found on PATH.\n"
            "Install it with:\n"
            "  Ubuntu/Debian : sudo apt install ffmpeg\n"
            "  macOS         : brew install ffmpeg\n"
            "  Windows       : https://ffmpeg.org/download.html"
        )


def _validate_inputs(audio_path: str, text_path: str, font_path: str) -> None:
    """Validate that all required input files exist.

    Args:
        audio_path: Path to the narration audio file.
        text_path: Path to the source text file.
        font_path: Path to the TrueType font file.

    Raises:
        SystemExit: If any file is missing.
    """
    for label, path in [("audio", audio_path), ("text", text_path), ("font", font_path)]:
        if not Path(path).exists():
            raise SystemExit(f"ERROR: {label} file not found: {path}")


def _cache_path_for(audio_path: str, text_path: str) -> str:
    """Derive the alignment cache file path from the input files.

    The cache file is placed alongside the audio file and named by the
    SHA-256 of the combined input paths and contents.

    Args:
        audio_path: Path to the audio file.
        text_path: Path to the text file.

    Returns:
        Path string for the ``.json`` alignment cache file.
    """
    key = Aligner.cache_key(audio_path, text_path)[:16]
    return str(Path(audio_path).with_suffix(f".align_{key}.json"))


def _run_alignment(
    audio_path: str,
    text_path: str,
    language: str,
    force: bool = False,
) -> List[SentenceTimestamp]:
    """Load alignment from cache or run WhisperX if cache is missing/stale.

    Args:
        audio_path: Path to the narration audio file.
        text_path: Path to the source text file.
        language: BCP-47 language code.
        force: If True, ignore the cache and re-run alignment.

    Returns:
        List of :class:`~aligner.SentenceTimestamp` objects.
    """
    cache_file = _cache_path_for(audio_path, text_path)
    key = Aligner.cache_key(audio_path, text_path)

    if not force:
        cached = Aligner.load_cache(cache_file, key)
        if cached is not None:
            logger.info(
                "Loaded %d sentence timestamps from cache: %s",
                len(cached),
                cache_file,
            )
            return cached

    logger.info("Running forced alignment (this may take several minutes) …")
    aligner = Aligner()
    text = Path(text_path).read_text(encoding="utf-8")
    timestamps = aligner.align(audio_path, text, language=language)
    Aligner.save_cache(timestamps, cache_file, key)
    return timestamps


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="video-tts",
        description=(
            "Generate an audio-synchronised scrolling-text video from a narration MP3 "
            "and its source text."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to the narration audio file (MP3 or WAV).",
    )
    parser.add_argument(
        "--text",
        required=True,
        metavar="PATH",
        help="Path to the source text file (.txt).",
    )
    parser.add_argument(
        "--output",
        default="video.mp4",
        metavar="PATH",
        help="Output MP4 video file path.",
    )
    parser.add_argument(
        "--font",
        default=DEFAULT_FONT_REGULAR,
        metavar="PATH",
        help="Path to the TrueType font file for body text.",
    )
    parser.add_argument(
        "--highlight_color",
        default="#ff4444",
        metavar="HEX",
        help="Hex colour for the active (currently spoken) sentence.",
    )
    parser.add_argument(
        "--bg_color",
        default="#1a1a1a",
        metavar="HEX",
        help="Background hex colour.",
    )
    parser.add_argument(
        "--scroll_speed_ms",
        default=300,
        type=int,
        metavar="MS",
        help="Scroll transition duration in milliseconds.",
    )
    parser.add_argument(
        "--language",
        default="en",
        metavar="CODE",
        help="BCP-47 language code for the alignment model.",
    )
    parser.add_argument(
        "--force_align",
        action="store_true",
        help="Ignore cached alignment data and re-run WhisperX.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Run the full video generation pipeline.

    Args:
        argv: Optional argument list for programmatic invocation.
            Defaults to ``sys.argv[1:]``.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    t_total = time.monotonic()

    # ── Step 1: Validate inputs ──────────────────────────────────────
    logger.info("[1/5] Validating inputs …")
    _check_ffmpeg()
    _validate_inputs(args.audio, args.text, args.font)
    logger.info(
        "  audio=%s  text=%s  output=%s",
        args.audio,
        args.text,
        args.output,
    )

    # ── Step 2: Forced alignment ─────────────────────────────────────
    logger.info("[2/5] Running forced alignment …")
    t_align = time.monotonic()
    timestamps = _run_alignment(
        audio_path=args.audio,
        text_path=args.text,
        language=args.language,
        force=args.force_align,
    )
    logger.info(
        "  %d sentence timestamps ready (%.1f s).",
        len(timestamps),
        time.monotonic() - t_align,
    )

    # ── Step 3: Build layout and line map ────────────────────────────
    logger.info("[3/5] Building layout and line map …")
    sentences = [ts.text for ts in timestamps]
    layout_engine = LayoutEngine(
        sentences=sentences,
        font_path=args.font,
    )
    logger.info("  %d lines in virtual canvas.", len(layout_engine.line_map))

    # ── Step 4: Initialise renderer ──────────────────────────────────
    logger.info("[4/5] Initialising frame renderer …")
    renderer = FrameRenderer(
        layout_engine=layout_engine,
        bg_color=args.bg_color,
        highlight_color=args.highlight_color,
    )

    # ── Step 5: Render frames and mux audio ─────────────────────────
    logger.info("[5/5] Rendering video …")
    t_render = time.monotonic()
    assembler = VideoAssembler(scroll_speed_ms=float(args.scroll_speed_ms))
    assembler.assemble(
        timestamps=timestamps,
        layout_engine=layout_engine,
        frame_renderer=renderer,
        audio_path=args.audio,
        output_path=args.output,
    )
    t_render_elapsed = time.monotonic() - t_render

    # ── Summary ──────────────────────────────────────────────────────
    total_duration_ms = max(ts.end_ms for ts in timestamps)
    output_size_mb = Path(args.output).stat().st_size / 1_048_576 if Path(args.output).exists() else 0

    from video_assembler import FPS, MS_PER_FRAME

    total_frames = int(total_duration_ms / MS_PER_FRAME)
    minutes, seconds = divmod(total_duration_ms / 1000, 60)

    print("\n" + "─" * 60)
    print("  Scrolling video generation complete")
    print("─" * 60)
    print(f"  Sentences          : {len(timestamps)}")
    print(f"  Total frames       : {total_frames}")
    print(f"  Video duration     : {int(minutes)}m {seconds:.1f}s")
    print(f"  Render time        : {t_render_elapsed:.1f}s")
    print(f"  Total time         : {time.monotonic() - t_total:.1f}s")
    print(f"  Output file        : {args.output}")
    print(f"  Output size        : {output_size_mb:.1f} MB")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
