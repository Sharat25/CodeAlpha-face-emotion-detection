"""

Data loading pipeline: directory -> tf.data.Dataset, with augmentation and
class-weight computation for the from-scratch CNN and the transfer-learning
model.

Both loaders share one contract: labels are integer indices into
config.CLASS_NAMES (alphabetical), and every image is normalized to [0, 1]
(or backbone-specific preprocessing for transfer learning) before it reaches
the model -- augmentation and normalization live in the data pipeline, never
duplicated in the model definitions.

"""

from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers
from deepfer import config

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation_layer() -> tf.keras.Sequential:

    """
    
    Rotation + scaling (zoom) + horizontal flip, exactly as specified in
    the project brief's 'Data Augmentation' requirement, plus a small random
    translation to make the model robust to imperfect face crops.
    
    """

    return tf.keras.Sequential(

        [

            layers.RandomFlip("horizontal") if config.AUG_HORIZONTAL_FLIP else layers.Layer(),

            layers.RandomRotation(config.AUG_ROTATION, fill_mode = "nearest"),

            layers.RandomZoom(config.AUG_ZOOM, fill_mode = "nearest"),

            layers.RandomTranslation(config.AUG_TRANSLATION, config.AUG_TRANSLATION, fill_mode = "nearest")

        ],

        name = "augmentation",

    )


def compute_class_weights(train_dir = None) -> dict:

    """
    
    Balanced class weights (sklearn 'balanced' formula:
    w_i = n_samples / (n_classes * n_i)) computed from actual file counts on
    disk. FER-2013 is heavily imbalanced (disgust has ~6% as many training
    images as happy) -- without this, the model just learns to rarely
    predict 'disgust' and still gets low loss.

    """

    train_dir = train_dir or config.TRAIN_DIR

    counts = Counter()

    for cls in config.CLASS_NAMES:

        cls_dir = train_dir / cls

        n = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        counts[cls] = n

    total = sum(counts.values())

    n_classes = len(config.CLASS_NAMES)

    weights = {i: total / (n_classes * counts[cls]) for i, cls in enumerate(config.CLASS_NAMES)}

    return weights


def _load_split(directory, image_size, color_mode, batch_size, shuffle):
    
    return tf.keras.utils.image_dataset_from_directory(

        directory,

        labels = "inferred",

        label_mode = "int",

        class_names = config.CLASS_NAMES,

        color_mode = color_mode,

        image_size = image_size,

        batch_size = batch_size,

        shuffle = shuffle,

        seed = config.SEED

    )


def get_scratch_datasets(batch_size: int = None):

    "Grayscale 48x48 pipeline for the from-scratch CNN."

    batch_size = batch_size or config.SCRATCH_BATCH_SIZE

    aug = build_augmentation_layer()

    train_ds = _load_split(config.TRAIN_DIR, config.SCRATCH_INPUT_SIZE, "grayscale", batch_size, shuffle = True)

    val_ds = _load_split(config.VAL_DIR, config.SCRATCH_INPUT_SIZE, "grayscale", batch_size, shuffle = False)

    test_ds = _load_split(config.TEST_DIR, config.SCRATCH_INPUT_SIZE, "grayscale", batch_size, shuffle = False)

    def prep_train(x, y):

        x = tf.cast(x, tf.float32) / 255.0

        x = aug(x, training = True)

        return x, y

    def prep_eval(x, y):

        x = tf.cast(x, tf.float32) / 255.0

        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    val_ds = val_ds.map(prep_eval, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    test_ds = test_ds.map(prep_eval, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, compute_class_weights()


def get_transfer_datasets(batch_size: int = None):
    
    """
    
    160x160 RGB pipeline for the transfer-learning model (resolution comes
    from config.TRANSFER_INPUT_SIZE, so this docstring's number must be kept
    in sync if that value changes again). Grayscale is
    replicated to 3 channels and run through MobileNetV2's own
    preprocess_input (scales to [-1, 1] -- NOT the same normalization as the
    scratch pipeline, which is why this is a separate function rather than a
    shared one with a flag).
    
    """
    
    batch_size = batch_size or config.TRANSFER_BATCH_SIZE
    
    aug = build_augmentation_layer()
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    train_ds = _load_split(config.TRAIN_DIR, config.TRANSFER_INPUT_SIZE, "grayscale", batch_size, shuffle = True)

    val_ds = _load_split(config.VAL_DIR, config.TRANSFER_INPUT_SIZE, "grayscale", batch_size, shuffle = False)

    test_ds = _load_split(config.TEST_DIR, config.TRANSFER_INPUT_SIZE, "grayscale", batch_size, shuffle = False)

    def to_rgb(x):

        return tf.image.grayscale_to_rgb(x)

    def prep_train(x, y):

        x = to_rgb(x)

        x = aug(x, training = True)

        x = preprocess_input(x)

        return x, y

    def prep_eval(x, y):

        x = to_rgb(x)

        x = preprocess_input(x)

        return x, y


    train_ds = train_ds.map(prep_train, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    val_ds = val_ds.map(prep_eval, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    test_ds = test_ds.map(prep_eval, num_parallel_calls = AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, compute_class_weights()