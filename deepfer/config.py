"""

Central configuration for the DeepFER project.

Keeping every path / hyperparameter in one importable module means train.py,
evaluate.py, optimize_export.py, realtime_webcam.py and the Flask app all
agree on class order, image size, and where artifacts live -- there is
exactly one place to change any of it.

"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = DATA_DIR / "train"

VAL_DIR = DATA_DIR / "val"

TEST_DIR = DATA_DIR / "test"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PLOTS_DIR = OUTPUTS_DIR / "plots"

LOGS_DIR = OUTPUTS_DIR / "logs"

METRICS_DIR = OUTPUTS_DIR / "metrics"


for d in (SAVED_MODELS_DIR, PLOTS_DIR, LOGS_DIR, METRICS_DIR):
    
    d.mkdir(parents = True, exist_ok = True)


# ---------------------------------------------------------------------------
# Data / classes
# ---------------------------------------------------------------------------

# Alphabetical order == the order Keras' image_dataset_from_directory infers
# from the class subfolder names. Every module MUST use this exact order,
# or predicted-index -> label-name mapping silently breaks.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

NUM_CLASSES = len(CLASS_NAMES)

# Emoji purely for the real-time overlay / web app, not used by the model.
CLASS_EMOJI = {

    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",

    "neutral": "😐", "sad": "😢", "surprise": "😲"

}

RAW_IMAGE_SIZE = (48, 48)  # native FER-2013 resolution, grayscale


# ---------------------------------------------------------------------------
# Scratch-CNN hyperparameters
# ---------------------------------------------------------------------------

SCRATCH_INPUT_SIZE = (48, 48)

SCRATCH_CHANNELS = 1  # grayscale, trained natively -- no need to fake RGB

SCRATCH_BATCH_SIZE = 64

SCRATCH_EPOCHS = 40

SCRATCH_LR = 1e-3


# ---------------------------------------------------------------------------
# Transfer-learning hyperparameters
# ---------------------------------------------------------------------------

# Pretrained ImageNet backbones expect 3-channel input at a reasonable
# resolution. We upsample 48x48 grayscale -> 160x160 RGB (replicate channel).
# 160x160 was chosen empirically: it reached 58.4% test accuracy versus
# 49.6% at 96x96 and offered a better accuracy/compute trade-off than
# MobileNetV2's native 224x224 (see README / report "Design Decisions" and
# the resolution ablation in outputs/metrics for the full comparison).
TRANSFER_INPUT_SIZE = (160, 160)

TRANSFER_CHANNELS = 3

TRANSFER_BACKBONE = "MobileNetV2"

TRANSFER_BATCH_SIZE = 64

HEAD_EPOCHS = 15         # phase 1: base frozen, train the classification head

FINE_TUNE_EPOCHS = 8     # phase 2: unfreeze top of the base, fine-tune end-to-end

HEAD_LR = 1e-3

FINE_TUNE_LR = 1e-5

FINE_TUNE_UNFREEZE_LAYERS = 30  # number of top backbone layers to unfreeze


# ---------------------------------------------------------------------------
# Augmentation (explicitly required by the project brief: rotation, scaling,
# flipping)
# ---------------------------------------------------------------------------

AUG_ROTATION = 0.10       # fraction of 2*pi -> ~36 degrees max

AUG_ZOOM = 0.15           # random scale +/-15%

AUG_HORIZONTAL_FLIP = True

AUG_TRANSLATION = 0.10    # small random shift, helps with imperfect face crops

SEED = 42