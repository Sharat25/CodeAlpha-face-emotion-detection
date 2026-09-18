"""

Shared evaluation + plotting helpers used by train.py, train_transfer.py
and evaluate.py, so the metrics reported for both models are computed
identically and are directly comparable.

"""
import json
import matplotlib
matplotlib.use("Agg")  # headless: this sandbox has no display server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score)
from deepfer import config


def collect_predictions(model, dataset):

    """
    
    Runs the model over a tf.data.Dataset and returns (y_true, y_pred, y_prob)
    as numpy arrays, aligned example-for-example.
    
    """

    y_true, y_prob = [], []
    
    for x_batch, y_batch in dataset:
        
        probs = model.predict(x_batch, verbose = 0)
        
        y_prob.append(probs)
        
        y_true.append(y_batch.numpy())
        
    y_true = np.concatenate(y_true)
    
    y_prob = np.concatenate(y_prob)
    
    y_pred = y_prob.argmax(axis = 1)
    
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, class_names=config.CLASS_NAMES) -> dict:

    acc = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average = "macro", zero_division = 0)

    recall_macro = recall_score(y_true, y_pred, average = "macro", zero_division = 0)

    f1_macro = f1_score(y_true, y_pred, average = "macro", zero_division = 0)

    precision_weighted = precision_score(y_true, y_pred, average = "weighted", zero_division = 0)

    recall_weighted = recall_score(y_true, y_pred, average = "weighted", zero_division = 0)

    f1_weighted = f1_score(y_true, y_pred, average = "weighted", zero_division = 0)
    
    per_class = classification_report(y_true, y_pred, target_names = class_names, output_dict = True, zero_division = 0)

    return {

        "accuracy": acc,

        "precision_macro": precision_macro,

        "recall_macro": recall_macro,

        "f1_macro": f1_macro,

        "precision_weighted": precision_weighted,

        "recall_weighted": recall_weighted,

        "f1_weighted": f1_weighted,

        "per_class": per_class

    }


def save_metrics(metrics: dict, path):

    path = str(path)

    with open(path, "w") as f:

        json.dump(metrics, f, indent = 2)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=True, title="Confusion Matrix"):

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:

        with np.errstate(all = "ignore"):

            cm_display = cm.astype(float) / cm.sum(axis = 1, keepdims = True)

            cm_display = np.nan_to_num(cm_display)

        fmt = ".2f"

    else:

        cm_display = cm

        fmt = "d"

    fig, ax = plt.subplots(figsize = (7, 6))

    im = ax.imshow(cm_display, cmap = "Blues", vmin = 0)

    ax.set_xticks(range(len(class_names)))

    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names, rotation = 45, ha = "right")

    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    ax.set_title(title)

    thresh = cm_display.max() / 2.0

    for i in range(cm_display.shape[0]):

        for j in range(cm_display.shape[1]):

            val = cm_display[i, j]

            ax.text(

                j, i, format(val, fmt),

                ha = "center", va = "center",

                color = "white" if val > thresh else "black",

                fontsize = 9

            )

    fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)

    fig.tight_layout()

    fig.savefig(out_path, dpi = 150)

    plt.close(fig)

    return cm


def plot_training_curves(history_csv_path, out_path, title="Training curves"):

    import pandas as pd

    df = pd.read_csv(history_csv_path)

    epoch_display = df["epoch"] + 1  # CSVLogger writes 0-indexed epochs; display as 1-indexed without touching the source file

    fig, axes = plt.subplots(1, 2, figsize = (11, 4))

    axes[0].plot(epoch_display, df["loss"], label = "train loss")

    axes[0].plot(epoch_display, df["val_loss"], label = "val loss")

    axes[0].set_xlabel("Epoch")

    axes[0].set_ylabel("Loss")

    axes[0].set_title("Loss")

    axes[0].legend()

    axes[0].grid(alpha = 0.3)

    axes[1].plot(epoch_display, df["accuracy"], label = "train acc")

    axes[1].plot(epoch_display, df["val_accuracy"], label = "val acc")

    axes[1].set_xlabel("Epoch")

    axes[1].set_ylabel("Accuracy")

    axes[1].set_title("Accuracy")

    axes[1].legend()

    axes[1].grid(alpha = 0.3)

    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig(out_path, dpi = 150)

    plt.close(fig)
