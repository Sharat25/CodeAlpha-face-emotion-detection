#!/usr/bin/env python3
"""

Evaluate a trained DeepFER checkpoint on the held-out TEST set (never seen
during training or validation-based checkpoint selection).

Usage
-----
    python evaluate.py --checkpoint saved_models/scratch_best.keras --kind scratch --tag scratch
    python evaluate.py --checkpoint saved_models/transfer_finetune_best.keras --kind transfer --tag transfer_finetuned

Writes:
    outputs/metrics/<tag>_test_metrics.json   accuracy, macro/weighted P/R/F1, per-class report
    outputs/plots/<tag>_confusion_matrix.png  normalized confusion matrix heatmap
    
"""

import argparse
import json
import time
import tensorflow as tf
from deepfer import config
from deepfer.dataset import get_scratch_datasets, get_transfer_datasets
from deepfer.metrics_utils import collect_predictions, compute_metrics, plot_confusion_matrix, save_metrics


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required = True)

    ap.add_argument("--kind", choices = ["scratch", "transfer"], required = True, help = "Which preprocessing pipeline to use (must match how the checkpoint was trained)")

    ap.add_argument("--tag", required = True, help = "Filename prefix for outputs")

    ap.add_argument("--batch-size", type = int, default = 64)

    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    
    model = tf.keras.models.load_model(args.checkpoint)

    if args.kind == "scratch":
        
        _, _, test_ds, _ = get_scratch_datasets(args.batch_size)
        
    else:
        
        _, _, test_ds, _ = get_transfer_datasets(args.batch_size)

    print("Running inference on the test set ...")

    t0 = time.time()

    y_true, y_pred, y_prob = collect_predictions(model, test_ds)

    elapsed = time.time() - t0

    n = len(y_true)

    print(f"Evaluated {n} test images in {elapsed:.1f}s ({elapsed / n * 1000:.2f} ms/image)")

    metrics = compute_metrics(y_true, y_pred)

    metrics["n_test_images"] = n

    metrics["checkpoint"] = args.checkpoint

    metrics["inference_seconds_total"] = elapsed

    metrics["inference_ms_per_image"] = elapsed / n * 1000

    print("\n=== Test-set results ===")

    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print(f"Macro Precision: {metrics['precision_macro']:.4f}")

    print(f"Macro Recall: {metrics['recall_macro']:.4f}")

    print(f"Macro F1: {metrics['f1_macro']:.4f}")

    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

    print("\nPer-class F1:")

    for cls in config.CLASS_NAMES:

        f1 = metrics["per_class"][cls]["f1-score"]

        support = metrics["per_class"][cls]["support"]

        print(f"  {cls:>9s}: F1 = {f1:.3f}  (support = {int(support)})")

    metrics_path = config.METRICS_DIR / f"{args.tag}_test_metrics.json"

    save_metrics(metrics, metrics_path)

    print(f"\nSaved metrics -> {metrics_path}")

    cm_path = config.PLOTS_DIR / f"{args.tag}_confusion_matrix.png"

    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, cm_path, title = f"DeepFER ({args.tag}) - Test Confusion Matrix")

    print(f"Saved confusion matrix -> {cm_path}")


if __name__ == "__main__":

    main()
