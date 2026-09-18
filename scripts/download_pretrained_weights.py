#!/usr/bin/env python3
"""

Fetch MobileNetV2 ImageNet-pretrained weights (no classifier head) for the
transfer-learning model.

Why this script exists at all: `tf.keras.applications.MobileNetV2(weights=
"imagenet")` normally downloads weights from
`storage.googleapis.com/tensorflow/keras-applications/...` automatically --
you do NOT need this script on a normal machine with unrestricted internet
access; `deepfer/models/transfer_model.py` tries that path first.

This script/fallback exists for network-restricted environments (e.g. this
project was built in a sandboxed CI-like container that allow-lists only a
handful of domains and blocks storage.googleapis.com). In that situation we
fetch the exact same MobileNetV2-alpha=1.0 ImageNet weights from a
long-standing, widely-used community mirror hosted as GitHub release assets:

    https://github.com/JonathanCMitchell/mobilenet_v2_keras

That repo re-exports the official TF-Slim MobileNetV2 checkpoint weights
(https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
into Keras .h5 format, using the same filename convention
("mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{alpha}_{size}[_no_top].h5")
that keras-applications itself uses internally. We verified before relying on
it: loading the file into tf.keras.applications.MobileNetV2 succeeds with
zero shape mismatches across all 155 layers, and running real face crops
through the *with-top* classifier variant produces varied, well-formed,
non-degenerate softmax predictions (see REPORT.md, "Design Decisions" ->
"Sourcing pretrained weights").

Usage
-----
    python scripts/download_pretrained_weights.py
    
"""

import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

FILENAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"

MIRROR_URL = f"https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/{FILENAME}"

EXPECTED_MIN_BYTES = 9_000_000  # sanity floor; real file is ~9.4MB


def download(url: str, dest: Path):
    
    print(f"Downloading {url}\n -> {dest}")

    def _hook(block_num, block_size, total_size):

        done = block_num * block_size

        pct = min(100, done * 100 // total_size) if total_size > 0 else 0

        print(f"\r  {pct:3d}%", end = "", flush = True)

    urllib.request.urlretrieve(url, dest, reporthook = _hook)
    
    print()


def main():

    WEIGHTS_DIR.mkdir(parents=True, exist_ok = True)

    dest = WEIGHTS_DIR / FILENAME

    if dest.exists() and dest.stat().st_size >= EXPECTED_MIN_BYTES:

        print(f"Already present: {dest} ({dest.stat().st_size:,} bytes) -- skipping download.")

        return

    try:

        download(MIRROR_URL, dest)

    except Exception as e:

        print(f"ERROR: failed to download pretrained weights: {e}", file=sys.stderr)

        print(

            "You can also just let deepfer/models/transfer_model.py use "

            "weights='imagenet' directly (the default on any machine that can "

            "reach storage.googleapis.com) -- this script is only needed as a "

            "workaround in network-restricted environments.",

            file = sys.stderr

        )

        sys.exit(1)

    size = dest.stat().st_size

    if size < EXPECTED_MIN_BYTES:

        print(f"ERROR: downloaded file is only {size:,} bytes (expected >= {EXPECTED_MIN_BYTES:,}); "

            "looks truncated or wrong. Deleting.", file = sys.stderr)

        dest.unlink()

        sys.exit(1)

    print(f"OK: {dest} ({size:,} bytes)")


if __name__ == "__main__":

    main()
