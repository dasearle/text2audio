#!/usr/bin/env python3
"""
text2audio - Convert text to speech using Kokoro TTS

A simple CLI tool that converts text files or pasted text to MP3/WAV audio
using the Kokoro TTS engine. Runs locally with no API costs.

Usage:
    python text2audio.py input.txt -o output.mp3
    python text2audio.py -i -o output.mp3
    python text2audio.py --list-voices
    python text2audio.py --play output.mp3
    python text2audio.py input.txt -o output.mp3 --play --loop 3
"""

import argparse
import re
import sys
import platform
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play as pydub_play

# Max characters per chunk - conservative to avoid phoneme overflow
MAX_CHUNK_CHARS = 150


def get_kokoro_model():
    """Initialize and return the Kokoro TTS model."""
    try:
        from kokoro_onnx import Kokoro
        model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        return model
    except Exception as e:
        print(f"Error loading Kokoro model: {e}")
        print("Make sure you have downloaded the model files.")
        print("Run: pip install kokoro-onnx")
        sys.exit(1)


def list_voices(model):
    """List available voices."""
    print("Available voices:")
    for voice in model.get_voices():
        print(f"  - {voice}")


def clean_text_for_speech(text):
    """Remove markdown and special characters that disrupt speech."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove markdown headers (# ## ### etc)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)  # ***bold italic***
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"___([^_]+)___", r"\1", text)  # ___bold italic___
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

    # Remove strikethrough
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove image syntax ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)

    # Remove bullet points and list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)

    # Remove remaining special characters that sound bad
    text = re.sub(r"[#*_~`<>|\\]", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)

    return text.strip()


def split_text_into_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """Split text into chunks, ensuring each chunk is under max_chars."""
    # First, split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []

    for sentence in sentences:
        # If sentence fits, try to combine with previous chunk
        if chunks and len(chunks[-1]) + len(sentence) + 1 <= max_chars:
            chunks[-1] += " " + sentence
        elif len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Sentence too long - split on punctuation first
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in parts:
                if chunks and len(chunks[-1]) + len(part) + 1 <= max_chars:
                    chunks[-1] += " " + part
                elif len(part) <= max_chars:
                    chunks.append(part)
                else:
                    # Still too long - split on words
                    words = part.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 <= max_chars:
                            current += (" " if current else "") + word
                        else:
                            if current:
                                chunks.append(current)
                            # Handle single words longer than max_chars
                            if len(word) > max_chars:
                                # Split very long words
                                for i in range(0, len(word), max_chars):
                                    chunks.append(word[i:i+max_chars])
                                current = ""
                            else:
                                current = word
                    if current:
                        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def text_to_audio(model, text, voice="af_heart"):
    """Convert text to audio using Kokoro TTS, handling long texts by chunking."""
    chunks = split_text_into_chunks(text)

    if len(chunks) == 1:
        samples, sample_rate = model.create(chunks[0], voice=voice)
        return samples, sample_rate

    print(f"  Text split into {len(chunks)} chunks for processing...")
    all_samples = []
    sample_rate = None

    for i, chunk in enumerate(chunks, 1):
        sys.stdout.write(f"\r  Processing chunk {i}/{len(chunks)}...")
        sys.stdout.flush()
        samples, sample_rate = model.create(chunk, voice=voice)
        all_samples.append(samples)
        pause = np.zeros(int(sample_rate * 0.3))
        all_samples.append(pause)

    print()

    if all_samples:
        all_samples = all_samples[:-1]

    return np.concatenate(all_samples), sample_rate


def save_audio(samples, sample_rate, output_path, format):
    """Save audio samples to file in the specified format."""
    if format == "wav":
        sf.write(str(output_path), samples, sample_rate)
    elif format == "mp3":
        temp_wav = output_path.with_suffix(".temp.wav")
        sf.write(str(temp_wav), samples, sample_rate)
        try:
            audio = AudioSegment.from_wav(str(temp_wav))
            audio.export(str(output_path), format="mp3")
        finally:
            temp_wav.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def play_audio(file_path, loop=1):
    """Play an audio file. loop=0 means infinite loop."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Audio file not found: {file_path}")
        sys.exit(1)

    suffix = file_path.suffix.lower()
    if suffix == ".wav":
        audio = AudioSegment.from_wav(str(file_path))
    elif suffix == ".mp3":
        audio = AudioSegment.from_mp3(str(file_path))
    else:
        print(f"Error: Unsupported audio format: {suffix}")
        sys.exit(1)

    try:
        if loop == 0:
            print(f"Playing {file_path} on infinite loop (Ctrl+C to stop)...")
            count = 1
            while True:
                sys.stdout.write(f"\r  Loop {count}...")
                sys.stdout.flush()
                pydub_play(audio)
                count += 1
        elif loop == 1:
            print(f"Playing {file_path}...")
            pydub_play(audio)
        else:
            print(f"Playing {file_path} ({loop} times)...")
            for i in range(1, loop + 1):
                sys.stdout.write(f"\r  Playing {i}/{loop}...")
                sys.stdout.flush()
                pydub_play(audio)
            print()
    except KeyboardInterrupt:
        print("\nPlayback stopped.")


def read_interactive_input():
    """Read text from stdin in interactive mode."""
    print("Enter text to convert (press Ctrl+D when done):")
    print("-" * 40)
    try:
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        return "\n".join(lines)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert text to speech using Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text2audio.py input.txt -o output.mp3
  python text2audio.py input.txt -o output.wav -f wav
  python text2audio.py -i -o output.mp3
  python text2audio.py --list-voices

Playback examples:
  python text2audio.py --play output.mp3
  python text2audio.py --play output.wav --loop 3
  python text2audio.py --play output.mp3 --loop 0    # infinite loop
  python text2audio.py input.txt -o out.wav --play   # convert and play

Note: MP3 output requires ffmpeg installed on your system.
  macOS:  brew install ffmpeg
  Linux:  sudo apt install ffmpeg
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Input text file to convert (or audio file with --play)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output audio file path"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["mp3", "wav"],
        help="Output format (default: inferred from output filename, or mp3)"
    )
    parser.add_argument(
        "-v", "--voice",
        default="af_heart",
        help="Voice to use (default: af_heart)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode: paste text directly"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    parser.add_argument(
        "-p", "--play",
        nargs="?",
        const=True,
        metavar="FILE",
        help="Play audio file, or play output after conversion"
    )
    parser.add_argument(
        "-l", "--loop",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to loop playback (0 = infinite)"
    )

    args = parser.parse_args()

    # Play-only mode: just play an existing audio file
    if args.play and args.play is not True:
        play_audio(args.play, args.loop)
        return
    
    if args.play is True and args.input_file and args.input_file.suffix.lower() in [".mp3", ".wav"]:
        play_audio(args.input_file, args.loop)
        return

    # List voices mode
    if args.list_voices:
        print("Loading Kokoro TTS model...")
        model = get_kokoro_model()
        list_voices(model)
        return

    # Conversion mode
    print("Loading Kokoro TTS model...")
    model = get_kokoro_model()

    if args.interactive:
        text = read_interactive_input()
    elif args.input_file:
        if not args.input_file.exists():
            print(f"Error: Input file not found: {args.input_file}")
            sys.exit(1)
        text = args.input_file.read_text()
    else:
        parser.print_help()
        print("\nError: Please provide an input file or use -i for interactive mode.")
        sys.exit(1)

    if not text.strip():
        print("Error: No text to convert.")
        sys.exit(1)

    if not args.output:
        if args.input_file:
            args.output = args.input_file.with_suffix(".mp3")
        else:
            args.output = Path("output.mp3")

    if args.format:
        output_format = args.format
    elif args.output.suffix.lower() in [".mp3", ".wav"]:
        output_format = args.output.suffix.lower().lstrip(".")
    else:
        output_format = "mp3"

    if not args.output.suffix.lower() == f".{output_format}":
        args.output = args.output.with_suffix(f".{output_format}")

    text = clean_text_for_speech(text)
    print(f"Converting text ({len(text)} characters) using voice '{args.voice}'...")
    try:
        samples, sample_rate = text_to_audio(model, text, args.voice)
    except Exception as e:
        print(f"Error during TTS conversion: {e}")
        sys.exit(1)

    print(f"Saving to {args.output}...")
    try:
        save_audio(samples, sample_rate, args.output, output_format)
    except Exception as e:
        print(f"Error saving audio: {e}")
        if output_format == "mp3":
            print("Hint: Make sure ffmpeg is installed for MP3 output.")
        sys.exit(1)

    duration = len(samples) / sample_rate
    print(f"Done! Created {args.output} ({duration:.1f} seconds)")

    # Play after conversion if requested
    if args.play:
        play_audio(args.output, args.loop)


if __name__ == "__main__":
    main()
