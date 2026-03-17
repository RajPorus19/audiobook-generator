"""
main.py — CLI entry point for the audiobook TTS engine.

Usage examples:
    # Simplest: just pass the text file — produces narration.mp3 and video.mp4
    python main.py book.txt

    # With voice cloning:
    python main.py book.txt --output narration.mp3 --speaker_wav voice.wav

    # Custom speed and language:
    python main.py book.txt --language en --speed 0.9 --output slow.mp3
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import video_main

import numpy as np
from tqdm import tqdm  # type: ignore[import]

from audio_assembler import AudioAssembler
from text_processor import ChunkType, TextChunk, TextProcessor
from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _read_input(input_arg: str) -> str:
    """Load text from a file path or return it directly if it is inline text.

    Args:
        input_arg: A file path (must end in ``.txt``) or a raw text string.

    Returns:
        The text content to synthesise.

    Raises:
        FileNotFoundError: If a ``.txt`` path is given but does not exist.
    """
    path = Path(input_arg)
    if path.suffix.lower() == ".txt" or path.exists():
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_arg}")
        logger.info("Reading input file: %s", path)
        return path.read_text(encoding="utf-8")
    # Treat as an inline string.
    logger.info("Using inline input text (%d characters).", len(input_arg))
    return input_arg


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="audiobook-tts",
        description="Convert text to a human-like audiobook narration using Qwen3-TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="PATH_OR_TEXT",
        help="Path to a .txt file, or a quoted inline text string.",
    )
    parser.add_argument(
        "--output",
        default="narration.mp3",
        metavar="PATH",
        help="Output audio file path.  Extension determines format (.mp3 or .wav).",
    )
    parser.add_argument(
        "--speaker_wav",
        default=None,
        metavar="PATH",
        help="Path to a 6–30 second reference WAV for voice cloning (optional).",
    )
    parser.add_argument(
        "--language",
        default="en",
        metavar="CODE",
        help="BCP-47 language code, e.g. 'en', 'fr', 'de'.",
    )
    parser.add_argument(
        "--speed",
        default=1.0,
        type=float,
        metavar="MULTIPLIER",
        help="Speech rate multiplier in [0.7, 1.3].  1.0 = normal speed.",
    )
    parser.add_argument(
        "--export_wav",
        action="store_true",
        help="Also export a WAV file alongside the MP3.",
    )
    parser.add_argument(
        "--max_tokens",
        default=250,
        type=int,
        metavar="N",
        help="Maximum whitespace-token count per synthesis chunk.",
    )
    return parser


def synthesise_chunks(
    engine: TTSEngine,
    chunks: List[TextChunk],
) -> List[np.ndarray]:
    """Synthesise all text chunks, showing a tqdm progress bar.

    Args:
        engine: Initialised :class:`~tts_engine.TTSEngine`.
        chunks: Ordered list of :class:`~text_processor.TextChunk` objects.

    Returns:
        List of float32 NumPy audio arrays, one per chunk.
    """
    audio_chunks: List[np.ndarray] = []
    with tqdm(total=len(chunks), desc="Synthesising", unit="chunk") as pbar:
        for chunk in chunks:
            audio = engine.synthesize(chunk.text)
            audio_chunks.append(audio)
            pbar.update(1)
            pbar.set_postfix(tokens=chunk.token_count)
    return audio_chunks


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point — parse arguments and run the full TTS pipeline.

    Args:
        argv: Optional argument list for programmatic invocation (e.g. tests).
            Defaults to ``sys.argv[1:]``.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    # 1. Load input text                                                   #
    # ------------------------------------------------------------------ #
    raw_text = _read_input(args.input)
    logger.info("Input: %d characters.", len(raw_text))

    # ------------------------------------------------------------------ #
    # 2. Process text into chunks                                          #
    # ------------------------------------------------------------------ #
    processor = TextProcessor(max_tokens=args.max_tokens)
    chunks = processor.process(raw_text)
    logger.info("Text split into %d chunk(s).", len(chunks))
    if not chunks:
        logger.error("No text chunks produced — nothing to synthesise.")
        return

    # ------------------------------------------------------------------ #
    # 3. Initialise TTS engine (loads model weights once)                  #
    # ------------------------------------------------------------------ #
    engine = TTSEngine(
        speaker_wav=args.speaker_wav,
        language=args.language,
        speed=args.speed,
    )
    logger.info("TTS engine ready.  Sample rate: %d Hz.", engine.sample_rate)

    # ------------------------------------------------------------------ #
    # 4. Synthesise all chunks                                             #
    # ------------------------------------------------------------------ #
    t_start = time.monotonic()
    audio_chunks = synthesise_chunks(engine, chunks)
    t_synth = time.monotonic() - t_start

    # ------------------------------------------------------------------ #
    # 5. Assemble and export                                               #
    # ------------------------------------------------------------------ #
    assembler = AudioAssembler(sample_rate=engine.sample_rate)
    chunk_types = [c.chunk_type for c in chunks]
    final_audio = assembler.assemble(audio_chunks, chunk_types)

    export_wav = args.export_wav or Path(args.output).suffix.lower() == ".wav"
    mp3_path, wav_path = assembler.export(
        final_audio,
        args.output,
        export_wav=export_wav,
    )

    # ------------------------------------------------------------------ #
    # 6. Print synthesis stats                                             #
    # ------------------------------------------------------------------ #
    total_duration_s = len(final_audio) / engine.sample_rate
    minutes, seconds = divmod(total_duration_s, 60)
    print("\n" + "─" * 60)
    print("  Audiobook synthesis complete")
    print("─" * 60)
    print(f"  Chunks processed   : {len(chunks)}")
    print(f"  Audio duration     : {int(minutes)}m {seconds:.1f}s")
    print(f"  Processing time    : {t_synth:.1f}s")
    print(f"  Realtime factor    : {total_duration_s / t_synth:.2f}x")
    print(f"  Output (MP3)       : {mp3_path}")
    if wav_path:
        print(f"  Output (WAV)       : {wav_path}")
    print("─" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # 7. Generate video                                                    #
    # ------------------------------------------------------------------ #
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".txt" and input_path.exists():
        video_output = str(Path(mp3_path).with_name("video.mp4"))
        logger.info("Starting video generation → %s", video_output)
        video_main.main([
            "--audio", mp3_path,
            "--text", str(input_path),
            "--output", video_output,
            "--language", args.language,
        ])


if __name__ == "__main__":
    main()
