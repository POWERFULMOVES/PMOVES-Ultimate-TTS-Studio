#!/usr/bin/env python3
"""Test TTS audio synthesis with validation.

This script synthesizes actual audio and validates the output,
going beyond just model loading to verify end-to-end functionality.

The Gradio unified TTS endpoint has 92 parameters. This script provides
all required values based on the API recorder format.

Usage:
    python tools/test_synthesis.py [--url URL] [--engine ENGINE] [--output DIR]

Options:
    --url URL       Gradio server URL (default: http://127.0.0.1:7860/)
    --engine ENGINE TTS engine to test (default: KittenTTS)
    --output DIR    Output directory for audio files (default: /tmp/tts-test)

Note: The Gradio app also supports MCP server mode (53 tools).
Enable with GRADIO_MCP_SERVER=true or launch(mcp_server=True).
"""

import argparse
import os
import sys
import shutil
import wave

try:
    from gradio_client import Client
except ImportError:
    print("ERROR: gradio_client required. Install with: pip install gradio_client")
    sys.exit(1)

# Default URL - can be overridden via --url or GRADIO_URL env var
DEFAULT_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860/")

# Engines that work without reference audio
SIMPLE_ENGINES = {
    "KittenTTS": {"voice": "expr-voice-2-f"},
    "Kokoro TTS": {"voice": "af_heart", "speed": 1.0},
}

# Test phrases
TEST_PHRASES = [
    "Hello from PMOVES.",
    "Audio test complete.",
]


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f" {text}")
    print("=" * 50)


def validate_wav(filepath: str) -> dict:
    """Validate WAV file and return metadata."""
    result = {
        "valid": False,
        "size": 0,
        "duration": 0,
        "sample_rate": 0,
        "channels": 0,
        "errors": [],
    }

    if not os.path.exists(filepath):
        result["errors"].append("File does not exist")
        return result

    result["size"] = os.path.getsize(filepath)
    if result["size"] < 100:
        result["errors"].append(f"File too small ({result['size']} bytes)")
        return result

    try:
        with wave.open(filepath, "rb") as wf:
            result["sample_rate"] = wf.getframerate()
            result["channels"] = wf.getnchannels()
            frames = wf.getnframes()
            result["duration"] = frames / result["sample_rate"]

            if result["duration"] < 0.1:
                result["errors"].append(f"Duration too short ({result['duration']:.2f}s)")
            elif result["sample_rate"] not in [16000, 22050, 24000, 44100, 48000]:
                result["errors"].append(f"Unusual sample rate ({result['sample_rate']})")
            else:
                result["valid"] = True

    except wave.Error as e:
        result["errors"].append(f"Invalid WAV: {e}")
    except Exception as e:
        result["errors"].append(f"Error reading file: {e}")

    return result


def test_synthesis(
    client: Client,
    engine: str,
    text: str,
    output_dir: str,
    voice_params: dict,
) -> bool:
    """Test synthesis for a single engine/text combination."""
    print(f"\n  Testing: {engine}")
    print(f"  Text: {text[:50]}...")

    try:
        # Use keyword arguments with only essential params
        # Gradio will use defaults for all other 92 parameters
        kwargs = {
            "text_input": text,
            "tts_engine": engine,
            "audio_format": "wav",
        }

        # Add engine-specific voice selection
        if engine == "KittenTTS":
            kwargs["kitten_voice"] = voice_params.get("voice", "expr-voice-2-f")
        elif engine == "Kokoro TTS":
            kwargs["kokoro_voice"] = voice_params.get("voice", "af_heart")
            kwargs["kokoro_speed"] = voice_params.get("speed", 1.0)

        result = client.predict(**kwargs, api_name="/generate_unified_tts")

        # Check result
        if not result:
            print("  ❌ No result returned")
            return False

        audio_path = result[0] if isinstance(result, tuple) else result
        if not audio_path:
            print("  ❌ No audio path in result")
            return False

        # Validate the audio file
        validation = validate_wav(audio_path)

        if validation["valid"]:
            # Copy to output directory
            safe_name = text[:30].replace(" ", "_").replace(".", "") + ".wav"
            output_path = os.path.join(output_dir, f"{engine.replace(' ', '_')}_{safe_name}")
            shutil.copy(audio_path, output_path)

            print(f"  ✓ Generated: {validation['duration']:.2f}s @ {validation['sample_rate']}Hz")
            print(f"  ✓ Saved to: {output_path}")
            return True
        else:
            print(f"  ❌ Validation failed: {', '.join(validation['errors'])}")
            return False

    except Exception as e:
        print(f"  ❌ Synthesis error: {str(e)[:60]}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test TTS audio synthesis")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Gradio server URL")
    parser.add_argument("--engine", type=str, default="KittenTTS", help="TTS engine to test")
    parser.add_argument("--output", type=str, default="/tmp/tts-test", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Test all simple engines")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"
    output_dir = args.output

    print_header("TTS Audio Synthesis Test")
    print(f"Target: {url}")
    print(f"Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Connect to Gradio
    print("\nConnecting to Gradio...")
    try:
        client = Client(url, verbose=False)
        print("Connected successfully")
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}")
        return 1

    # Determine engines to test
    if args.all:
        engines_to_test = list(SIMPLE_ENGINES.keys())
    else:
        if args.engine not in SIMPLE_ENGINES:
            print(f"ERROR: Engine '{args.engine}' not in simple test list.")
            print(f"Available: {', '.join(SIMPLE_ENGINES.keys())}")
            return 1
        engines_to_test = [args.engine]

    # Test each engine
    results = {}
    for engine in engines_to_test:
        print_header(f"Testing {engine}")

        voice_params = SIMPLE_ENGINES[engine]
        success_count = 0

        for text in TEST_PHRASES:
            if test_synthesis(client, engine, text, output_dir, voice_params):
                success_count += 1

        results[engine] = success_count == len(TEST_PHRASES)
        print(f"\n  Result: {success_count}/{len(TEST_PHRASES)} phrases succeeded")

    # Summary
    print_header("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Engines tested: {total}")
    print(f"Passed: {passed}/{total}")

    for engine, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {engine}")

    print(f"\nAudio files saved to: {output_dir}")
    print("Done!")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
