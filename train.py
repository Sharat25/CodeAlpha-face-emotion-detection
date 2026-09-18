#!/usr/bin/env python3
"""

Train the from-scratch DeepFER CNN on FER-2013.

Usage
-----
    python train.py --epochs 40 --batch-size 64 --lr 1e-3

What this script does, in order:
1. Builds train/val/test tf.data pipelines (deepfer.dataset) with augmentation on the train split only.
2. Computes balanced class weights from the actual on-disk file counts (FER-2013's 'disgust' class has ~6% as many images as 'happy').
3. Builds deepfer.models.cnn_scratch and compiles it with Adam + a LR schedule.
4. Trains with:
    - ModelCheckpoint: saves the best epoch by val_accuracy
    - EarlyStopping: stops if val_loss stalls (restores best weights)
    - ReduceLROnPlateau: halves LR when val_loss plateaus
    - CSVLogger: per-epoch metrics -> outputs/logs/scratch_history.csv
5. Writes training-curve plots and a final summary JSON.

Run `python evaluate.py --model scratch` afterwards for test-set metrics
(precision/recall/F1/confusion matrix) -- this script only reports
train/val numbers, exactly like an ML training run should (the test set
must stay untouched until final evaluation).

"""
import argparse
import json
import time
import tensorflow as tf
from deepfer import config
from deepfer.dataset import get_scratch_datasets
from deepfer.metrics_utils import plot_training_curves
from deepfer.models.cnn_scratch import build_scratch_cnn


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--epochs", type = int, default = config.SCRATCH_EPOCHS)

    ap.add_argument("--batch-size", type = int, default = config.SCRATCH_BATCH_SIZE)

    ap.add_argument("--lr", type = float, default = config.SCRATCH_LR)

    ap.add_argument("--patience", type=int, default = 8, help = "EarlyStopping patience (epochs)")

    ap.add_argument("--no-class-weights", action = "store_true", help = "Disable balanced class weighting")

    ap.add_argument("--tag", default = "scratch", help = "Filename prefix for checkpoints/logs")

    ap.add_argument(
        
        "--resume", action = "store_true",help = "Resume from saved_models/<tag>_last.keras if it exists (for chunked "
        
        "training across multiple bounded runs -- see README 'Reproducing training')."
        
    )

    ap.add_argument(
        
        "--max-seconds", type = float, default = None,
    
        help = "Soft wall-clock budget for this invocation. A callback stops training "
    
        "(cleanly, saving checkpoints) once elapsed time exceeds this, even mid-epoch-schedule, "
    
        "instead of relying on the caller to guess how many epochs fit in the time available."
                        
    )

    args = ap.parse_args()

    tf.random.set_seed(config.SEED)

    print("Loading datasets ...")

    train_ds, val_ds, test_ds, class_weights = get_scratch_datasets(args.batch_size)

    print("Class weights (balanced, computed from train-set file counts):")

    for i, name in enumerate(config.CLASS_NAMES):

        print(f"  {name:>9s}: {class_weights[i]:.3f}")

    ckpt_best = config.SAVED_MODELS_DIR / f"{args.tag}_best.keras"

    ckpt_last = config.SAVED_MODELS_DIR / f"{args.tag}_last.keras"

    history_csv = config.LOGS_DIR / f"{args.tag}_history.csv"

    initial_epoch = 0

    if args.resume and ckpt_last.exists():

        print(f"Resuming from {ckpt_last}")

        model = tf.keras.models.load_model(ckpt_last)

        if history_csv.exists():

            import pandas as pd

            initial_epoch = len(pd.read_csv(history_csv))

        print(f"Resuming at epoch {initial_epoch + 1}")

    else:

        model = build_scratch_cnn()

        model.compile(

            optimizer = tf.keras.optimizers.Adam(learning_rate = args.lr),

            loss = "sparse_categorical_crossentropy",

            metrics = ["accuracy"]

        )

        model.summary()


    class TimeBudget(tf.keras.callbacks.Callback):

        """
        
        Stops training after --max-seconds elapses, so one bounded tool
        call always ends cleanly (with checkpoints saved) instead of being
        killed mid-write by the outer sandbox timeout.
        
        """
        
        def __init__(self, max_seconds):

            super().__init__()

            self.max_seconds = max_seconds

            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):

            if self.max_seconds and (time.time() - self.t0) > self.max_seconds:

                print(f"\n[TimeBudget] {self.max_seconds}s elapsed -> stopping after epoch {epoch + 1}")

                self.model.stop_training = True

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(str(ckpt_best), monitor = "val_accuracy", mode = "max", save_best_only = True, verbose = 1),

        tf.keras.callbacks.ModelCheckpoint(str(ckpt_last), save_best_only = False, verbose = 0),

        tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = args.patience, restore_best_weights = True, verbose = 1),

        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1),

        tf.keras.callbacks.CSVLogger(str(history_csv), append = args.resume)

    ]
    
    if args.max_seconds:
        
        callbacks.append(TimeBudget(args.max_seconds))

    t0 = time.time()

    history = model.fit(

        train_ds,

        validation_data = val_ds,

        initial_epoch = initial_epoch,

        epochs = args.epochs,

        class_weight = None if args.no_class_weights else class_weights,

        callbacks = callbacks,

        verbose = 2

    )
    
    elapsed = time.time() - t0

    plot_path = config.PLOTS_DIR / f"{args.tag}_training_curves.png"

    plot_training_curves(history_csv, plot_path, title="Scratch CNN - Training Curves")

    best_val_acc = max(history.history["val_accuracy"])

    summary = {

        "model": "scratch_cnn",

        "epochs_ran": len(history.history["loss"]),

        "epochs_requested": args.epochs,

        "best_val_accuracy": best_val_acc,

        "final_train_accuracy": history.history["accuracy"][-1],

        "training_seconds": elapsed,

        "batch_size": args.batch_size,

        "initial_lr": args.lr,

        "class_weighted": not args.no_class_weights,

        "checkpoint": str(ckpt_best)

    }

    with open(config.METRICS_DIR / f"{args.tag}_train_summary.json", "w") as f:

        json.dump(summary, f, indent=2)

    print("\n=== Training complete ===")

    print(json.dumps(summary, indent = 2))

    print(f"Best checkpoint: {ckpt_best}")

    print(f"Training curves: {plot_path}")


if __name__ == "__main__":
    
    main()
