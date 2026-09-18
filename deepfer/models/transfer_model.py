"""

Transfer-learning model: MobileNetV2 (ImageNet-pretrained) as a frozen
feature extractor, topped with a small classification head, then
selectively fine-tuned.

Why MobileNetV2 specifically: it is the lightest widely-used ImageNet
backbone (~2.3M conv-backbone params via depthwise-separable convolutions),
which matters a lot when the only available compute is a single CPU core --
a ResNet50/VGG16 backbone would be several times slower per image for a
modest accuracy difference on a 7-class, low-resolution problem like this.

Two-phase strategy (both phases live in train_transfer.py, this module only builds the architecture):
Phase 1 - feature extraction: backbone frozen, only the head trains.
            Cheap enough to precompute backbone features once and train the
            head on cached arrays for many fast epochs.
Phase 2 - fine-tuning: unfreeze the top FINE_TUNE_UNFREEZE_LAYERS layers
            of the backbone and continue training end-to-end at a much
            lower learning rate, so the high-level ImageNet features adapt
            to facial expressions without destroying the pretrained
            low-level filters.

"""

import tensorflow as tf
from tensorflow.keras import layers, models
from deepfer import config

LOCAL_WEIGHTS_FALLBACK = (config.PROJECT_ROOT / "weights" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5")


def build_backbone(input_shape = (*config.TRANSFER_INPUT_SIZE, config.TRANSFER_CHANNELS)):
    
    """
    
    Loads MobileNetV2 pretrained on ImageNet.

    Tries the standard Keras auto-download first -- this is what runs on any
    normal machine with unrestricted internet access, and is the right code
    path 99% of the time. If that fails (e.g. a network-restricted sandbox
    that cannot reach storage.googleapis.com, which is the environment this
    project was actually developed and trained in) it falls back to a local
    copy of the exact same weight file. See
    scripts/download_pretrained_weights.py for how that local copy was
    obtained and verified.
    
    """
    try:

        backbone = tf.keras.applications.MobileNetV2(input_shape = input_shape, include_top = False, weights = "imagenet", pooling = None)

        print("[transfer_model] loaded ImageNet weights via the standard Keras auto-download.")

    except Exception as e:

        if not LOCAL_WEIGHTS_FALLBACK.exists():

            raise RuntimeError(

                "Could not auto-download ImageNet weights, and no local fallback was found at "
                
                f"{LOCAL_WEIGHTS_FALLBACK}.\nRun: python scripts/download_pretrained_weights.py"
                
            ) from e

        print(f"[transfer_model] standard ImageNet auto-download failed ({e.__class__.__name__}); " f"loading cached weights from {LOCAL_WEIGHTS_FALLBACK} instead.")

        backbone = tf.keras.applications.MobileNetV2(input_shape = input_shape, include_top = False, weights = None, pooling = None)

        backbone.load_weights(str(LOCAL_WEIGHTS_FALLBACK))

    backbone.trainable = False

    return backbone


def build_transfer_model(

    input_shape = (*config.TRANSFER_INPUT_SIZE, config.TRANSFER_CHANNELS),

    num_classes = config.NUM_CLASSES,

    dropout = 0.4

):

    backbone = build_backbone(input_shape)


    inputs = layers.Input(shape = input_shape, name = "image")

    x = backbone(inputs, training = False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation = "relu")(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation = "softmax", name = "emotion")(x)

    model = models.Model(inputs, outputs, name="deepfer_transfer_mobilenetv2")

    return model, backbone


def set_fine_tune_mode(model: tf.keras.Model, backbone: tf.keras.Model, n_unfreeze: int = config.FINE_TUNE_UNFREEZE_LAYERS):

    """
    
    Unfreeze the top `n_unfreeze` layers of the backbone in place, leaving
    everything below frozen. BatchNorm layers are kept frozen even when
    "unfrozen" positionally -- letting BN running stats drift on a small,
    imbalanced FER-2013 batch is a well-known way to destabilize fine-tuning
    of MobileNet-family models.
    
    """

    backbone.trainable = True

    freeze_until = max(0, len(backbone.layers) - n_unfreeze)

    for layer in backbone.layers[:freeze_until]:

        layer.trainable = False

    for layer in backbone.layers[freeze_until:]:

        if isinstance(layer, layers.BatchNormalization):

            layer.trainable = False

        else:

            layer.trainable = True

    return model


if __name__ == "__main__":

    model, backbone = build_transfer_model()

    model.summary()

    print(f"\nTotal params: {model.count_params():,}")

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

    print(f"Trainable params (phase 1, backbone frozen): {trainable:,}")
