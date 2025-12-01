# text2audio

A simple CLI tool to convert text files to speech using Kokoro TTS. Runs entirely locally with no API costs.

## Features

- Human-like voice synthesis using Kokoro TTS
- 54 voices across multiple languages (American, British, Japanese, Chinese, and more)
- Supports WAV and MP3 output formats
- Handles unlimited text length (automatic chunking)
- Cleans markdown formatting for natural speech
- Built-in audio player with loop support
- Interactive mode for pasting text directly
- Cross-platform (macOS and Linux)
- Completely free and unlimited - runs locally

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dasearle/text2audio.git
cd text2audio
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files

```bash
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 5. Install ffmpeg (optional, for MP3 output)

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

## Usage

**Important:** Always activate the virtual environment first:

```bash
source venv/bin/activate
```

### Command Line Options

```
usage: text2audio.py [-h] [-o OUTPUT] [-f {mp3,wav}] [-v VOICE] [-i]
                     [--list-voices] [-p [FILE]] [-l N]
                     [input_file]

positional arguments:
  input_file              Input text file to convert (or audio file with --play)

options:
  -h, --help              Show help message and exit
  -o, --output OUTPUT     Output audio file path
  -f, --format {mp3,wav}  Output format (default: inferred from filename, or mp3)
  -v, --voice VOICE       Voice to use (default: af_heart)
  -i, --interactive       Interactive mode: paste text directly
  --list-voices           List available voices and exit
  -p, --play [FILE]       Play audio file, or play output after conversion
  -l, --loop N            Number of times to loop playback (0 = infinite)
```

### Conversion Examples

```bash
# Convert a text file to WAV
python text2audio.py input.txt -o output.wav

# Convert to MP3 (requires ffmpeg)
python text2audio.py input.txt -o output.mp3

# Use a specific voice
python text2audio.py input.txt -o output.wav -v am_adam

# Interactive mode - paste text directly, then press Ctrl+D when done
python text2audio.py -i -o output.wav

# List all available voices
python text2audio.py --list-voices
```

### Playback Examples

```bash
# Play an audio file
python text2audio.py --play output.wav

# Play 3 times
python text2audio.py --play output.wav --loop 3

# Play on infinite loop (Ctrl+C to stop)
python text2audio.py --play output.wav --loop 0

# Convert and play immediately
python text2audio.py input.txt -o output.wav --play

# Convert and play 5 times
python text2audio.py input.txt -o output.wav --play --loop 5
```

## Available Voices

Run `python text2audio.py --list-voices` to see all 54 available voices. Voice prefixes indicate:

- `af_` / `am_` - American English (female/male)
- `bf_` / `bm_` - British English (female/male)
- `jf_` / `jm_` - Japanese (female/male)
- `zf_` / `zm_` - Chinese (female/male)
- And more...

Default voice is `af_heart`.

## Requirements

- Python 3.10+
- ~340MB disk space for model files
- ffmpeg (only for MP3 output and playback)

## License

MIT
