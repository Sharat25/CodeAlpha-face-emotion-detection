#!/usr/bin/env python3
"""

Train the MobileNetV2 transfer-learning model on FER-2013, in two phases.

    Phase "head": backbone frozen, only the classification head trains.
    Phase "finetune":  top N backbone layers unfrozen, trained end-to-end at
                        a much lower LR, initialized from the head-trained
                        checkpoint.

Usage
-----
    python train_transfer.py --phase head     --epochs 15
    python train_transfer.py --phase finetune --epochs 20

Like train.py, both phases support --resume/--max-seconds so a long run can
be split across several bounded invocations without losing progress --
every completed epoch's checkpoint is saved before the next one starts.

"""

import argparse
import json
import time
import tensorflow as tf
from deepfer import config
from deepfer.dataset import get_transfer_datasets
from deepfer.metrics_utils import plot_training_curves
from deepfer.models.transfer_model import build_transfer_model, set_fine_tune_mode


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--phase", choices = ["head", "finetune"], required = True)

    ap.add_argument("--epochs", type=int, default = None)

    ap.add_argument("--batch-size", type = int, default = config.TRANSFER_BATCH_SIZE)

    ap.add_argument("--lr", type=float, default = None)

    ap.add_argument("--patience", type = int, default = 10)

    ap.add_argument("--no-class-weights", action = "store_true")

    ap.add_argument("--tag", default = "transfer")

    ap.add_argument("--resume", action = "store_true")

    ap.add_argument("--max-seconds", type = float, default = None)

    args = ap.parse_args()

    epochs = (args.epochs or (config.HEAD_EPOCHS if args.phase == "head" else config.FINE_TUNE_EPOCHS))

    lr = args.lr or (config.HEAD_LR if args.phase == "head" else config.FINE_TUNE_LR)

    tf.random.set_seed(config.SEED)

    print("Loading datasets ...")

    train_ds, val_ds, test_ds, class_weights = get_transfer_datasets(args.batch_size)

    ckpt_best = config.SAVED_MODELS_DIR / f"{args.tag}_{args.phase}_best.keras"

    ckpt_last = config.SAVED_MODELS_DIR / f"{args.tag}_{args.phase}_last.keras"

    history_csv = config.LOGS_DIR / f"{args.tag}_{args.phase}_history.csv"

    initial_epoch = 0

    if args.resume and ckpt_last.exists():

        print(f"Resuming from {ckpt_last}")

        model = tf.keras.models.load_model(ckpt_last)

        if history_csv.exists() and history_csv.stat().st_size > 0:

            import pandas as pd

            try:

                initial_epoch = len(pd.read_csv(history_csv))

            except pd.errors.EmptyDataError:

                initial_epoch = 0

        print(f"Resuming at epoch {initial_epoch + 1}")

        epochs += initial_epoch

    elif args.phase == "head":

        model, backbone = build_transfer_model()

        model.compile(

            optimizer = tf.keras.optimizers.Adam(learning_rate = lr),

            loss = "sparse_categorical_crossentropy",

            metrics = ["accuracy"]

        )

        model.summary()
        
    else:  # finetune, fresh start from the best head checkpoint

        head_ckpt = config.SAVED_MODELS_DIR / f"{args.tag}_head_best.keras"

        if not head_ckpt.exists():

            raise SystemExit(f"No head checkpoint at {head_ckpt} -- run --phase head first.")

        print(f"Initializing fine-tune phase from {head_ckpt}")

        model = tf.keras.models.load_model(head_ckpt)

        # The backbone is the second layer of the functional model (index 1: input, backbone, GAP, ...).

        backbone_layer = next(l for l in model.layers if isinstance(l, tf.keras.Model))

        set_fine_tune_mode(model, backbone_layer, n_unfreeze = config.FINE_TUNE_UNFREEZE_LAYERS)

        model.compile(

            optimizer=tf.keras.optimizers.Adam(learning_rate = lr),

            loss = "sparse_categorical_crossentropy",

            metrics = ["accuracy"]

        )

        n_trainable = sum(w.shape.num_elements() for w in model.trainable_weights)

        print(
            
            f"Unfroze top {config.FINE_TUNE_UNFREEZE_LAYERS} backbone layers"

            f"({n_trainable:,} trainable params) for fine-tuning at lr = {lr}"

        )


    class TimeBudget(tf.keras.callbacks.Callback):

        def __init__(self, max_seconds):

            super().__init__()

            self.max_seconds = max_seconds

            self.t0 = time.time()


        def on_epoch_end(self, epoch, logs = None):

            if self.max_seconds and (time.time() - self.t0) > self.max_seconds:

                print(f"\n[TimeBudget] {self.max_seconds}s elapsed -> stopping after epoch {epoch + 1}")

                self.model.stop_training = True


    callbacks = [
        
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_best), monitor = "val_accuracy", mode = "max", save_best_only = True, verbose = 1),

        tf.keras.callbacks.ModelCheckpoint(str(ckpt_last), save_best_only = False, verbose = 0),

        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = args.patience, restore_best_weights = True, verbose = 1),

        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7, verbose = 1),

        tf.keras.callbacks.CSVLogger(str(history_csv), append = args.resume)
    ]
    
    if args.max_seconds:
        
        callbacks.append(TimeBudget(args.max_seconds))

    t0 = time.time()
    
    history = model.fit(

        train_ds,

        validation_data = val_ds,

        initial_epoch = initial_epoch,

        epochs = epochs,

        class_weight = None if args.no_class_weights else class_weights,

        callbacks = callbacks,

        verbose = 2

    )

    elapsed = time.time() - t0

    plot_path = config.PLOTS_DIR / f"{args.tag}_{args.phase}_training_curves.png"

    plot_training_curves(history_csv, plot_path, title = f"Transfer Model ({args.phase}) - Training Curves")

    summary = {

        "model": f"transfer_mobilenetv2_{args.phase}",

        "epochs_ran": len(history.history["loss"]),

        "epochs_requested": epochs,

        "best_val_accuracy": max(history.history["val_accuracy"]) if history.history.get("val_accuracy") else None,

        "training_seconds_this_invocation": elapsed,

        "batch_size": args.batch_size,

        "lr": lr,

        "checkpoint": str(ckpt_best)

    }
    
    with open(config.METRICS_DIR / f"{args.tag}_{args.phase}_train_summary.json", "w") as f:

        json.dump(summary, f, indent = 2)

    print("\n=== Training complete ===")

    print(json.dumps(summary, indent = 2))


if __name__ == "__main__":
    
    main()