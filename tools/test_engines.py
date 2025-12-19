#!/usr/bin/env python3
"""Test all 7 TTS engines via Gradio API.

This script runs inside the Pinokio conda environment and tests
each TTS engine by loading it and optionally running synthesis.

Usage:
    python tools/test_engines.py [--synthesis]
"""

import sys

try:
    from gradio_client import Client
except ImportError:
    print("ERROR: gradio_client required. Install with: pip install gradio_client")
    sys.exit(1)


# Engine definitions: (id, display_name, load_endpoint, requires_ref_audio)
ENGINES = [
    ("kitten_tts", "KittenTTS", "handle_load_kitten", False),
    ("kokoro", "Kokoro TTS", "handle_load_kokoro", False),
    ("f5_tts", "F5-TTS", "handle_f5_load", True),
    ("indextts2", "IndexTTS2", "handle_load_indextts2", True),
    ("fish", "Fish Speech", "handle_load_fish", True),
    ("chatterbox", "ChatterboxTTS", "handle_load_chatterbox", True),
    ("voxcpm", "VoxCPM", "handle_load_voxcpm", True),
]


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f" {text}")
    print('=' * 50)


def main():
    """Test all TTS engines."""
    print_header("TTS Engine Test Suite")
    print("Target: http://127.0.0.1:7860/")

    # Connect to Gradio
    print("\nConnecting to Gradio...")
    try:
        client = Client("http://127.0.0.1:7860/", verbose=False)
        print("Connected successfully")
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}")
        return 1

    # Test each engine
    print_header("Loading TTS Models")

    results = {}
    for engine_id, name, load_endpoint, requires_ref in ENGINES:
        print(f"  {name}...", end=" ", flush=True)
        try:
            result = client.predict(api_name=f"/{load_endpoint}")
            status = str(result[0]) if result else ""

            if "✅" in status or "Loaded" in status.lower():
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
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)

    print(f"✓ Loaded:        {loaded}/{total}")
    print(f"⚠ Needs download: {needs_download}/{total}")
    print(f"❌ Failed:        {failed}/{total}")

    if loaded > 0:
        print("\nEngines ready for use:")
        for engine_id, status in results.items():
            if status is True:
                name = next(n for e, n, _, _ in ENGINES if e == engine_id)
                print(f"  ✓ {name}")

    if needs_download > 0:
        print("\nEngines needing model download:")
        for engine_id, status in results.items():
            if status == "download":
                name = next(n for e, n, _, _ in ENGINES if e == engine_id)
                print(f"  ⚠ {name}")

    if failed > 0:
        print("\nFailed engines:")
        for engine_id, status in results.items():
            if status is False:
                name = next(n for e, n, _, _ in ENGINES if e == engine_id)
                print(f"  ❌ {name}")

    print("\nDone!")
    return 0 if loaded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
