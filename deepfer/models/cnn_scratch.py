"""

A CNN designed from scratch for 48x48 grayscale facial-emotion classification
(no pretrained weights -- contrast this with deepfer/models/transfer_model.py).

Design rationale (see REPORT for the full writeup):
  * Four convolutional blocks with progressively more filters (32 -> 64 ->
    128 -> 256) let the network build from low-level edges/textures up to
    higher-level facial-expression features.
  * Two stacked 3x3 convs per block (blocks 1-2) approximate a larger
    receptive field with fewer parameters than one 5x5/7x7 conv, standard
    practice since VGG.
  * BatchNorm after every conv stabilizes and speeds up training -- with a
    single CPU core and no GPU, faster convergence per-epoch matters a lot.
  * GlobalAveragePooling2D instead of Flatten+big-Dense before the head
    collapses the final 3x3x256 feature map to a 256-vector with ZERO extra
    parameters (versus ~1.2M for a naive Flatten->Dense(256)). This is a
    direct response to FER-2013 being small and badly imbalanced (only 392
    training images for 'disgust') -- fewer parameters in the head means
    less overfitting on the rare classes.
  * Dropout before the final classifier for additional regularization.
  
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from deepfer import config


def conv_block(x, filters, n_convs = 2, l2 = 1e-4):

    for _ in range(n_convs):

        x = layers.Conv2D(

            filters, 3, padding = "same", use_bias = False,

            kernel_regularizer = regularizers.l2(l2),

        )(x)

        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

    return x


def build_scratch_cnn(input_shape = (*config.SCRATCH_INPUT_SIZE, config.SCRATCH_CHANNELS), num_classes = config.NUM_CLASSES, dropout = 0.5) -> tf.keras.Model:
  
    inputs = layers.Input(shape = input_shape, name = "image")

    x = conv_block(inputs, 32, n_convs = 2)
        
    x = layers.MaxPooling2D()(x)              # 48 -> 24
    
    
    x = conv_block(x, 64, n_convs = 2)
        
    x = layers.MaxPooling2D()(x)              # 24 -> 12
        
    x = layers.Dropout(0.25)(x)
    
    
    x = conv_block(x, 128, n_convs = 2)
        
    x = layers.MaxPooling2D()(x)              # 12 -> 6
        
    x = layers.Dropout(0.25)(x)
    
    
    x = conv_block(x, 256, n_convs = 1)
        
    x = layers.MaxPooling2D()(x)              # 6 -> 3
    

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation = "relu", kernel_regularizer=regularizers.l2(1e-4))(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation = "softmax", name = "emotion")(x)
    
    return models.Model(inputs, outputs, name="deepfer_scratch_cnn")


if __name__ == "__main__":

    m = build_scratch_cnn()

    m.summary()

    print(f"\nTotal params: {m.count_params():,}")
