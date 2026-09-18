"""

prepare_dataset.py
-------------------
Turns the raw FER-2013 archive (as distributed on Kaggle: train/<class>/*.jpg,
test/<class>/*.jpg) into a clean, reproducible train / val / test split on disk:

    data/processed/train/<class>/*.jpg
    data/processed/val/<class>/*.jpg
    data/processed/test/<class>/*.jpg

Why a script instead of doing the split on the fly in the Dataset loader:
  * The split is stratified and seeded, so it is IDENTICAL every run -- anyone
    re-running this on the same raw data gets the same train/val boundary.
    Splitting on the fly with a fresh random seed per run would silently let
    validation images leak into training between runs, which invalidates any
    "best model" checkpoint selection based on val accuracy.
  * Physically separating the files means later code (Keras
    image_dataset_from_directory, manual PIL loaders, the Flask app's demo
    image picker, etc.) can all just point at a directory -- no custom split
    logic duplicated in five places.

Usage
-----
    python scripts/prepare_dataset.py \
        --raw-dir /path/to/archive-3 \
        --out-dir data/processed \
        --val-fraction 0.1 \
        --seed 42

`--raw-dir` must contain `train/` and `test/` subfolders, each with one
subfolder per emotion class (the standard Kaggle FER-2013 "archive" layout).

"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def collect_images(class_dir: Path):

    exts = {".jpg", ".jpeg", ".png"}

    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in exts)


def stratified_split(files, val_fraction, seed):

    rng = random.Random(seed)

    files = list(files)

    rng.shuffle(files)

    n_val = max(1, round(len(files) * val_fraction)) if files else 0

    return files[n_val:], files[:n_val]  # train, val


def copy_all(files, dest_dir: Path):

    dest_dir.mkdir(parents = True, exist_ok = True)

    for f in files:

        shutil.copy2(f, dest_dir / f.name)


def main():

    ap = argparse.ArgumentParser(description = __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--raw-dir", required = True, help = "Path to extracted FER-2013 folder containing train/ and test/")

    ap.add_argument("--out-dir", default = "data/processed", help = "Where to write the organized split")

    ap.add_argument("--val-fraction", type = float, default = 0.10, help = "Fraction of the ORIGINAL train split held out for validation")

    ap.add_argument("--seed", type = int, default = 42)

    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)

    out_dir = Path(args.out_dir)

    raw_train = raw_dir / "train"

    raw_test = raw_dir / "test"

    if not raw_train.is_dir() or not raw_test.is_dir():

        raise SystemExit(

            f"Expected '{raw_dir}' to contain train/ and test/ subfolders "

            f"(the standard FER-2013 Kaggle layout). Found: "

            f"{[p.name for p in raw_dir.iterdir() if p.is_dir()]}"

        )

    stats = defaultdict(dict)
    
    for cls in CLASS_NAMES:

        cls_dir = raw_train / cls

        if not cls_dir.is_dir():

            raise SystemExit(f"Missing expected class folder: {cls_dir}")

        all_train_files = collect_images(cls_dir)

        train_files, val_files = stratified_split(all_train_files, args.val_fraction, args.seed)

        test_cls_dir = raw_test / cls

        test_files = collect_images(test_cls_dir) if test_cls_dir.is_dir() else []

        copy_all(train_files, out_dir / "train" / cls)

        copy_all(val_files, out_dir / "val" / cls)

        copy_all(test_files, out_dir / "test" / cls)

        stats[cls] = {"train": len(train_files), "val": len(val_files), "test": len(test_files)}

        print(f"{cls:>9s}  train = {len(train_files):5d}  val = {len(val_files):4d}  test = {len(test_files):4d}")

    totals = {split: sum(stats[c][split] for c in CLASS_NAMES) for split in ("train", "val", "test")}

    print(f"{'TOTAL':>9s}  train = {totals['train']:5d}  val = {totals['val']:4d}  test = {totals['test']:4d}")

    meta = {

        "class_names": CLASS_NAMES,

        "val_fraction_of_train": args.val_fraction,

        "seed": args.seed,

        "counts": stats,

        "totals": totals

    }

    out_dir.mkdir(parents = True, exist_ok = True)

    with open(out_dir / "dataset_stats.json", "w") as f:

        json.dump(meta, f, indent=2)

    print(f"\nWrote split + stats to {out_dir.resolve()}")


if __name__ == "__main__":
    
    main()
