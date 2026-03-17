"""
Video Assembler — streams frames to an MP4 and muxes the original audio.

Memory model:
  Frames are written to an OpenCV VideoWriter one at a time.  At most one frame
  is held in memory simultaneously, so even a 10-hour audiobook at 30 FPS
  (1 080 000 frames × ~6 MB each uncompressed) never exceeds a few hundred MB.

Scroll interpolation:
  - When a new sentence activates the active sentence index changes and
    a scroll transition is started from the current interpolated position
    to the new scroll target.
  - Mid-transition interruption is handled by re-targeting from the
    current position — no jump, no queued transitions.
  - Cubic ease-in-out gives a natural deceleration feel.

Audio mux:
  FFmpeg is called as a subprocess to copy the video stream and encode the
  MP3 audio as AAC into the final container.  The original video frames are
  left untouched (``-c:v copy``).
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2  # type: ignore[import]
import numpy as np
from tqdm import tqdm  # type: ignore[import]

from aligner import SentenceTimestamp
from frame_renderer import FrameRenderer
from layout_engine import LayoutEngine

logger = logging.getLogger(__name__)

FPS: int = 30
MS_PER_FRAME: float = 1000.0 / FPS  # ≈ 33.33 ms


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out easing function.

    Maps a normalised time value in [0, 1] to a smoothed progress value
    using the standard CSS ``ease-in-out`` cubic curve.

    Args:
        t: Normalised time, clamped to [0, 1].

    Returns:
        Eased progress value in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t ** 3
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


class _ScrollState:
    """Tracks the scroll position and active scroll transition.

    Attributes:
        current_y: Current interpolated scroll position in virtual-canvas pixels.
        target_y: Destination scroll position.
        start_y: Scroll position when the current transition began.
        transition_start_ms: Audio-relative time when the transition began.
        transition_duration_ms: Duration of the current transition.
    """

    def __init__(self, initial_y: float = 0.0, transition_ms: float = 300.0) -> None:
        """Initialise scroll state.

        Args:
            initial_y: Starting scroll position.
            transition_ms: Default transition duration in milliseconds.
        """
        self.current_y: float = initial_y
        self.target_y: float = initial_y
        self.start_y: float = initial_y
        self.transition_start_ms: float = 0.0
        self.transition_duration_ms: float = transition_ms

    def start_transition(self, target_y: float, now_ms: float) -> None:
        """Begin a new scroll transition, interrupting any in-progress one.

        Args:
            target_y: New scroll destination.
            now_ms: Current audio timestamp in milliseconds.
        """
        self.start_y = self.current_y       # Start from wherever we currently are.
        self.target_y = target_y
        self.transition_start_ms = now_ms

    def update(self, now_ms: float) -> None:
        """Advance the scroll interpolation to the current time.

        Args:
            now_ms: Current audio timestamp in milliseconds.
        """
        elapsed = now_ms - self.transition_start_ms
        if elapsed >= self.transition_duration_ms:
            self.current_y = self.target_y
        else:
            t = elapsed / self.transition_duration_ms
            eased = ease_in_out_cubic(t)
            self.current_y = self.start_y + (self.target_y - self.start_y) * eased


def _get_audio_duration_ms(audio_path: str) -> int:
    """Return the duration of an audio file in milliseconds via ffprobe.

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.).

    Returns:
        Duration in milliseconds.  Returns 0 on any error.
    """
    import json as _json
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format=duration",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = _json.loads(result.stdout)
        return int(float(data["format"]["duration"]) * 1000)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read audio duration: %s", exc)
        return 0


def _check_ffmpeg() -> None:
    """Verify that the FFmpeg system binary is available on PATH.

    Raises:
        RuntimeError: If ``ffmpeg`` is not found.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg binary not found on PATH.\n"
            "Install it with:  sudo apt install ffmpeg  (Debian/Ubuntu)\n"
            "                  brew install ffmpeg       (macOS)"
        )


def _find_active_sentence(
    timestamps: List[SentenceTimestamp],
    current_ms: float,
) -> int:
    """Return the index of the sentence active at ``current_ms``.

    A sentence is active when ``start_ms <= current_ms < end_ms``.
    If no sentence is active, the most recently started sentence is returned.

    Args:
        timestamps: Ordered list of sentence timestamps.
        current_ms: Current audio time in milliseconds.

    Returns:
        Zero-based sentence index in ``timestamps``.
    """
    last_started = 0
    for ts in timestamps:
        if ts.start_ms <= current_ms:
            last_started = ts.index
        if ts.start_ms <= current_ms < ts.end_ms:
            return ts.index
    return last_started


class VideoAssembler:
    """Orchestrates frame rendering, video writing, and audio muxing.

    The assembler streams frames directly to an OpenCV VideoWriter, keeping
    memory usage constant regardless of video duration.  After all frames are
    written, FFmpeg is used to mux the original audio into the final MP4.

    Example::

        assembler = VideoAssembler()
        assembler.assemble(
            timestamps=timestamps,
            layout_engine=layout_engine,
            frame_renderer=renderer,
            audio_path="narration.mp3",
            output_path="video.mp4",
        )
    """

    def __init__(
        self,
        fps: int = FPS,
        scroll_speed_ms: float = 300.0,
    ) -> None:
        """Initialise the assembler.

        Args:
            fps: Output frame rate.
            scroll_speed_ms: Duration of each scroll transition in milliseconds.
        """
        _check_ffmpeg()
        self._fps = fps
        self._ms_per_frame = 1000.0 / fps
        self._scroll_speed_ms = scroll_speed_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        timestamps: List[SentenceTimestamp],
        layout_engine: LayoutEngine,
        frame_renderer: FrameRenderer,
        audio_path: str,
        output_path: str,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> None:
        """Render all frames, write to a temp file, then mux with audio.

        Args:
            timestamps: Sentence timing data from the aligner.
            layout_engine: Provides scroll targets via ``get_scroll_target``.
            frame_renderer: Callable via ``render(scroll_y, active_sentence_idx)``.
            audio_path: Path to the source narration audio file.
            output_path: Destination MP4 file path.
            video_width: Frame width in pixels.
            video_height: Frame height in pixels.

        Raises:
            RuntimeError: If FFmpeg is unavailable or the mux step fails.
        """
        if not timestamps:
            raise ValueError("Cannot assemble video: timestamps list is empty.")

        # Use the actual audio file duration so the video always matches the
        # audio length — alignment timestamps may be shorter than the audio if
        # the text only partially covers it.
        audio_duration_ms = _get_audio_duration_ms(audio_path)
        total_duration_ms = max(max(ts.end_ms for ts in timestamps), audio_duration_ms)
        total_frames = int(np.ceil(total_duration_ms / self._ms_per_frame))

        logger.info(
            "Rendering %d frames at %d FPS (%.1f s) …",
            total_frames,
            self._fps,
            total_duration_ms / 1000.0,
        )

        # Write frames to a temporary silent MP4.
        with tempfile.NamedTemporaryFile(suffix="_silent.mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._write_frames(
                total_frames=total_frames,
                timestamps=timestamps,
                layout_engine=layout_engine,
                frame_renderer=frame_renderer,
                tmp_path=tmp_path,
                video_width=video_width,
                video_height=video_height,
            )
            self._mux_audio(
                silent_video=tmp_path,
                audio_path=audio_path,
                output_path=output_path,
            )
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)

        logger.info("Video assembled: %s", output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_frames(
        self,
        total_frames: int,
        timestamps: List[SentenceTimestamp],
        layout_engine: LayoutEngine,
        frame_renderer: FrameRenderer,
        tmp_path: str,
        video_width: int,
        video_height: int,
    ) -> None:
        """Render and write all frames to a silent MP4 file.

        Frames are rendered one at a time and handed to the VideoWriter
        immediately — no frame buffer is accumulated.

        Args:
            total_frames: Total number of frames to render.
            timestamps: Sentence timing data.
            layout_engine: Provides scroll targets.
            frame_renderer: Provides rendered frames.
            tmp_path: Destination path for the silent MP4.
            video_width: Frame width.
            video_height: Frame height.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            tmp_path,
            fourcc,
            float(self._fps),
            (video_width, video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {tmp_path}")

        scroll = _ScrollState(
            initial_y=layout_engine.get_scroll_target(0),
            transition_ms=self._scroll_speed_ms,
        )
        prev_active = -1

        t_render_start = time.monotonic()
        frames_written = 0

        try:
            with tqdm(total=total_frames, desc="Rendering frames", unit="frame") as pbar:
                for frame_idx in range(total_frames):
                    current_ms = frame_idx * self._ms_per_frame
                    active_idx = _find_active_sentence(timestamps, current_ms)

                    # Start a new scroll transition when the active sentence changes.
                    if active_idx != prev_active:
                        new_target = layout_engine.get_scroll_target(active_idx)
                        scroll.start_transition(new_target, current_ms)
                        prev_active = active_idx

                    scroll.update(current_ms)
                    frame_bgr = frame_renderer.render(scroll.current_y, active_idx)
                    writer.write(frame_bgr)
                    frames_written += 1
                    pbar.update(1)
        finally:
            writer.release()

        elapsed = time.monotonic() - t_render_start
        logger.info(
            "Rendered %d frames in %.1f s (%.1f FPS effective).",
            frames_written,
            elapsed,
            frames_written / elapsed if elapsed > 0 else 0,
        )

    @staticmethod
    def _mux_audio(
        silent_video: str,
        audio_path: str,
        output_path: str,
    ) -> None:
        """Combine the silent video and the original audio into the final MP4.

        Uses FFmpeg with ``-c:v copy`` to avoid re-encoding the video stream,
        and ``-c:a aac`` for broad MP4 compatibility.

        Args:
            silent_video: Path to the silent MP4 file.
            audio_path: Path to the narration audio file.
            output_path: Destination path for the final MP4.

        Raises:
            RuntimeError: If FFmpeg exits with a non-zero status.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",                      # Overwrite output without prompt.
            "-i", silent_video,
            "-i", audio_path,
            "-c:v", "copy",            # Copy video stream unchanged.
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",               # Trim to shorter of video / audio.
            output_path,
        ]
        logger.info("Muxing audio: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg mux failed (exit {result.returncode}):\n{result.stderr}"
            )
        logger.info("Audio mux complete → %s", output_path)
