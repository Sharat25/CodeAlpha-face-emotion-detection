#!/usr/bin/env python3
"""
Performance Optimization (project brief requirement #6): convert a trained
Keras model to TFLite, apply post-training quantization, and benchmark the
real accuracy/latency/size trade-off between:
    1. Baseline Keras (.keras) model, float32
    2. TFLite, float32 (graph-optimized, no quantization)
    3. TFLite, dynamic-range quantization (int8 weights, float32 activations)
    4. TFLite, full integer quantization (int8 weights AND activations, calibrated on a representative sample of real training images)

Every number in the printed table and the saved JSON is measured on this
machine, not estimated -- accuracy is recomputed per variant on the same
test-set subset, and latency is a real wall-clock average over N inferences
with the TFLite Interpreter API.

Usage
-----
    python optimize_export.py --checkpoint saved_models/scratch_best.keras --kind scratch
"""
import argparse
import json
import time
import numpy as np
import tensorflow as tf
from deepfer import config
from deepfer.dataset import get_scratch_datasets, get_transfer_datasets

N_LATENCY_RUNS = 100

N_ACCURACY_SAMPLES = 2000  # random (seeded) subset of the test set, for a fast-but-real accuracy readout per variant


def representative_dataset_gen(sample_images):

    def gen():

        for img in sample_images:

            yield [img[np.newaxis, ...].astype(np.float32)]

    return gen


def convert_tflite(keras_model, quantization: str, representative_images = None) -> bytes:

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if quantization == "float32":

        pass  # default graph optimizations only, no quantization

    elif quantization == "dynamic_range":

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization == "full_int8":

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.representative_dataset = representative_dataset_gen(representative_images)

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        converter.inference_input_type = tf.float32   # keep I/O float32 for a drop-in-compatible API

        converter.inference_output_type = tf.float32
        
    else:

        raise ValueError(quantization)

    return converter.convert()


def benchmark_tflite(tflite_bytes: bytes, images: np.ndarray, labels: np.ndarray):

    interpreter = tf.lite.Interpreter(model_content = tflite_bytes)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    output_details = interpreter.get_output_details()[0]

    # accuracy over N_ACCURACY_SAMPLES
    correct = 0

    for img, label in zip(images, labels):

        interpreter.set_tensor(input_details["index"], img[np.newaxis, ...].astype(input_details["dtype"]))

        interpreter.invoke()

        pred = interpreter.get_tensor(output_details["index"])[0]

        if np.argmax(pred) == label:

            correct += 1

    accuracy = correct / len(images)

    # latency over N_LATENCY_RUNS (separate loop, untimed accuracy pass excluded) 
    sample = images[0][np.newaxis, ...].astype(input_details["dtype"])

    for _ in range(5):  # warmup

        interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()

    t0 = time.time()

    for _ in range(N_LATENCY_RUNS):

        interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()

    latency_ms = (time.time() - t0) / N_LATENCY_RUNS * 1000

    return accuracy, latency_ms


def benchmark_keras(model, images: np.ndarray, labels: np.ndarray):

    probs = model.predict(images, verbose = 0, batch_size = 64)

    accuracy = float((probs.argmax(axis = 1) == labels).mean())

    sample = images[0:1]

    for _ in range(5):

        model.predict(sample, verbose = 0)

    t0 = time.time()

    for _ in range(N_LATENCY_RUNS):

        model.predict(sample, verbose=0)

    latency_ms = (time.time() - t0) / N_LATENCY_RUNS * 1000

    return accuracy, latency_ms


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--kind", choices=["scratch", "transfer"], required = True)

    ap.add_argument("--tag", default=None)

    args = ap.parse_args()

    tag = args.tag or args.kind

    model = tf.keras.models.load_model(args.checkpoint)

    if args.kind == "scratch":

        _, _, test_ds, _ = get_scratch_datasets(batch_size = 64)

    else:

        _, _, test_ds, _ = get_transfer_datasets(batch_size = 64)

    # Materialize a fixed-size numpy subset once, shared by every variant so
    # the accuracy/latency comparison is apples-to-apples. IMPORTANT: the
    # underlying test_ds is built with shuffle=False (deliberately, so
    # evaluate.py's full-test-set metrics are 100% reproducible) -- which
    # means it walks the directory in class order. Taking a naive "first N"
    # slice of that would silently sample almost exclusively from the first
    # 1-2 alphabetical classes. We shuffle here, locally, with a fixed seed,
    # specifically for drawing a representative subset -- this does NOT
    # affect evaluate.py or training.
    all_images, all_labels = [], []

    for xb, yb in test_ds:

        all_images.append(xb.numpy())

        all_labels.append(yb.numpy())

    all_images = np.concatenate(all_images)

    all_labels = np.concatenate(all_labels)

    rng = np.random.RandomState(config.SEED)

    n_sample = min(N_ACCURACY_SAMPLES, len(all_images))

    idx = rng.choice(len(all_images), size = n_sample, replace = False)

    images = all_images[idx]

    labels = all_labels[idx]

    print(
        
        f"Accuracy subset: {n_sample} images, shuffled, class distribution "

        f"{np.bincount(labels, minlength=config.NUM_CLASSES).tolist()} "

        f"(classes: {config.CLASS_NAMES})"
        
        )

    representative_images = images[:200]  # calibration subset for full-int8

    results = {}

    print(f"[1/4] Baseline Keras (.keras, float32) ...")

    acc, lat = benchmark_keras(model, images, labels)

    keras_size = sum(w.numpy().nbytes for w in model.weights)

    results["keras_float32"] = {"accuracy": acc, "latency_ms": lat, "size_bytes": keras_size}

    print(f"accuracy={acc:.4f}  latency={lat:.2f} ms  size={keras_size/1e6:.2f} MB (weights only)")


    for name, quant in [
        
        ("tflite_float32", "float32"),

        ("tflite_dynamic_range_int8", "dynamic_range"),

        ("tflite_full_int8", "full_int8")
                        
]:

        print(f"[.../4] {name} ...")

        tfl_bytes = convert_tflite(model, quant, representative_images)

        acc, lat = benchmark_tflite(tfl_bytes, images, labels)

        out_path = config.SAVED_MODELS_DIR / f"{tag}_{name}.tflite"

        with open(out_path, "wb") as f:

            f.write(tfl_bytes)

        results[name] = {"accuracy": acc, "latency_ms": lat, "size_bytes": len(tfl_bytes), "file": str(out_path)}

        print(f"accuracy={acc:.4f}  latency={lat:.2f} ms  size={len(tfl_bytes)/1e6:.2f} MB  -> {out_path}")


    baseline_lat = results["keras_float32"]["latency_ms"]

    baseline_size = results["keras_float32"]["size_bytes"]

    print(f"\n{'variant':<28s}{'accuracy':>10s}{'latency(ms)':>14s}{'speedup':>10s}{'size(MB)':>11s}{'size_reduction':>16s}")

    for name, r in results.items():

        speedup = baseline_lat / r["latency_ms"]

        size_reduction = 1 - r["size_bytes"] / baseline_size

        print(f"{name:<28s}{r['accuracy']:>10.4f}{r['latency_ms']:>14.2f}{speedup:>9.2f}x{r['size_bytes']/1e6:>10.2f}{size_reduction:>15.1%}")

    out_json = config.METRICS_DIR / f"{tag}_optimization_results.json"

    with open(out_json, "w") as f:

        json.dump({"n_accuracy_samples": len(images), "n_latency_runs": N_LATENCY_RUNS, "results": results}, f, indent=2)

    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    
    main()
