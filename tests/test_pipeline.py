"""

Sanity tests for the DeepFER pipeline. These are fast, structural checks
(shapes, ranges, param counts, label ordering) -- NOT a substitute for the
real accuracy numbers in outputs/metrics/, which come from evaluate.py on
the held-out test set. Their job is to catch the kind of bug that silently
corrupts results (see optimize_export.py's revision history / README for a
real example: a naive "first N" sample from an unshuffled directory listing
that accidentally evaluated on one class only).

Run with:  python -m pytest tests/ -v

"""
import numpy as np
import pytest
import tensorflow as tf
from deepfer import config
from deepfer.dataset import compute_class_weights, get_scratch_datasets
from deepfer.models.cnn_scratch import build_scratch_cnn


def test_class_names_alphabetical():

    """
    
    Every module trusts config.CLASS_NAMES to be alphabetical, matching

    what image_dataset_from_directory would infer on its own -- if this

    ever drifts, predicted class indices silently point at the wrong label.
    
    """

    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)

    assert len(config.CLASS_NAMES) == config.NUM_CLASSES == 7


def test_scratch_cnn_output_shape():

    model = build_scratch_cnn()

    x = tf.zeros((2, *config.SCRATCH_INPUT_SIZE, config.SCRATCH_CHANNELS))

    y = model(x, training=False)

    assert y.shape == (2, config.NUM_CLASSES)

    # softmax output: each row sums to ~1
    sums = tf.reduce_sum(y, axis = 1).numpy()

    np.testing.assert_allclose(sums, [1.0, 1.0], atol = 1e-4)


def test_class_weights_favor_rare_classes():

    weights = compute_class_weights()

    disgust_idx = config.CLASS_NAMES.index("disgust")

    happy_idx = config.CLASS_NAMES.index("happy")

    # disgust has ~6% as many training images as happy -> its balanced
    # weight must be substantially larger, or class imbalance isn't
    # actually being corrected.
    assert weights[disgust_idx] > 5 * weights[happy_idx]


@pytest.mark.slow
def test_dataset_pipeline_shapes_and_labels():

    """
    
    Loads a couple of real batches from disk -- requires
    data/processed/ to exist (run scripts/prepare_dataset.py first).
    
    """

    train_ds, val_ds, test_ds, _ = get_scratch_datasets(batch_size=8)

    xb, yb = next(iter(train_ds))

    assert xb.shape == (8, *config.SCRATCH_INPUT_SIZE, 1)

    assert xb.numpy().min() >= 0.0 and xb.numpy().max() <= 1.0

    assert yb.shape == (8,)

    assert set(np.unique(yb.numpy())).issubset(set(range(config.NUM_CLASSES)))


if __name__ == "__main__":

    import sys

    sys.exit(pytest.main([__file__, "-v"]))
