#!/usr/bin/env python3
"""
text2audio - Convert text to speech using Kokoro TTS

A simple CLI tool that converts text files or pasted text to MP3/WAV audio
using the Kokoro TTS engine. Runs locally with no API costs.

Usage:
    python text2audio.py input.txt -o output.mp3
    python text2audio.py -i -o output.mp3
    python text2audio.py --list-voices
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment


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


def split_text_into_chunks(text, max_chars=200):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in parts:
                if len(current_chunk) + len(part) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


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

Note: MP3 output requires ffmpeg installed on your system.
  macOS:  brew install ffmpeg
  Linux:  sudo apt install ffmpeg
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Input text file to convert"
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

    args = parser.parse_args()

    print("Loading Kokoro TTS model...")
    model = get_kokoro_model()

    if args.list_voices:
        list_voices(model)
        return

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


if __name__ == "__main__":
    main()
