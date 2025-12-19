#!/usr/bin/env python3
"""Test TTS engines via Gradio API.

This script runs inside the Pinokio conda environment and tests
each TTS engine by loading it and optionally running synthesis.

Usage:
    python tools/test_engines.py [--url URL]

Options:
    --url URL   Gradio server URL (default: http://127.0.0.1:7860/)
"""

import argparse
import os
import sys

try:
    from gradio_client import Client
except ImportError:
    print("ERROR: gradio_client required. Install with: pip install gradio_client")
    sys.exit(1)

# Default URL - can be overridden via --url or GRADIO_URL env var
DEFAULT_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860/")


# Engine definitions: (id, display_name, load_endpoint, load_args)
# load_args: None for no params, dict for named params, or "skip" to skip loading
ENGINES = [
    ("kitten_tts", "KittenTTS", "handle_load_kitten", None),
    ("kokoro", "Kokoro TTS", "handle_load_kokoro", None),
    ("f5_tts", "F5-TTS", "handle_f5_load", {"model_name": "F5-TTS Base"}),
    ("indextts", "IndexTTS", "handle_load_indextts", None),
    ("indextts2", "IndexTTS2", "handle_load_indextts2", None),
    ("fish", "Fish Speech", "handle_load_fish", None),
    ("chatterbox", "ChatterboxTTS", "handle_load_chatterbox", None),
    ("chatterbox_mtl", "Chatterbox Multilingual", "handle_load_chatterbox_multilingual", None),
    ("higgs", "Higgs Audio", "handle_load_higgs", None),
    ("voxcpm", "VoxCPM", "handle_load_voxcpm", None),
    # VibeVoice requires model path - skip in basic test
    ("vibevoice", "VibeVoice", "handle_vibevoice_load", "skip"),
]


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f" {text}")
    print('=' * 50)


def main():
    """Test all TTS engines."""
    parser = argparse.ArgumentParser(description="Test TTS engines")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Gradio server URL")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"

    print_header("TTS Engine Test Suite")
    print(f"Target: {url}")

    # Connect to Gradio
    print("\nConnecting to Gradio...")
    try:
        client = Client(url, verbose=False)
        print("Connected successfully")
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}")
        return 1

    # Test each engine
    print_header("Loading TTS Models")

    results = {}
    for engine_id, name, load_endpoint, load_args in ENGINES:
        print(f"  {name}...", end=" ", flush=True)

        # Skip engines that require complex setup
        if load_args == "skip":
            print("⏭ Skipped (requires setup)")
            results[engine_id] = "skip"
            continue

        try:
            if load_args is None:
                result = client.predict(api_name=f"/{load_endpoint}")
            else:
                result = client.predict(**load_args, api_name=f"/{load_endpoint}")

            status = str(result[0]) if result else ""

            if "✅" in status or "Loaded" in status.lower() or "ready" in status.lower():
                print("✓ Loaded")
                results[engine_id] = True
            elif "download" in status.lower():
                print("⚠ Needs download")
                results[engine_id] = "download"
            else:
                # Extract error message
                error = status.replace("❌", "").strip()[:40]
                print(f"❌ {error or 'Failed'}")
                results[engine_id] = False

        except Exception as e:
            print(f"❌ {str(e)[:40]}")
            results[engine_id] = False

    # Summary
    print_header("Summary")

    loaded = sum(1 for v in results.values() if v is True)
    needs_download = sum(1 for v in results.values() if v == "download")
    skipped = sum(1 for v in results.values() if v == "skip")
    failed = sum(1 for v in results.values() if v is False)
    tested = len(results) - skipped

    print(f"✓ Loaded:        {loaded}/{tested}")
    print(f"⚠ Needs download: {needs_download}/{tested}")
    print(f"❌ Failed:        {failed}/{tested}")
    if skipped > 0:
        print(f"⏭ Skipped:       {skipped}")

    # Helper to get engine name
    def get_name(eid):
        return next(n for e, n, _, _ in ENGINES if e == eid)

    if loaded > 0:
        print("\nEngines ready for use:")
        for engine_id, status in results.items():
            if status is True:
                print(f"  ✓ {get_name(engine_id)}")

    if needs_download > 0:
        print("\nEngines needing model download:")
        for engine_id, status in results.items():
            if status == "download":
                print(f"  ⚠ {get_name(engine_id)}")

    if failed > 0:
        print("\nFailed engines:")
        for engine_id, status in results.items():
            if status is False:
                print(f"  ❌ {get_name(engine_id)}")

    print("\nDone!")
    return 0 if loaded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
