import tensorflow as tf
from src.model.conv_block import DownBlock, UpBlock

# U-Net generator using residual blocks for down and up sampling
def build_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    down_stack = [
        DownBlock(filters=64, kernel_size=4, apply_norm=False),
        DownBlock(filters=128, kernel_size=4),
        DownBlock(filters=256, kernel_size=4),
        DownBlock(filters=512, kernel_size=4),
        DownBlock(filters=512, kernel_size=4),
        DownBlock(filters=512, kernel_size=4),
        DownBlock(filters=512, kernel_size=4),
        DownBlock(filters=512, kernel_size=4),
    ]

    up_stack = [
        UpBlock(filters=512, kernel_size=4, apply_dropout=True),
        UpBlock(filters=512, kernel_size=4, apply_dropout=True),
        UpBlock(filters=512, kernel_size=4, apply_dropout=True),
        UpBlock(filters=512, kernel_size=4),
        UpBlock(filters=256, kernel_size=4),
        UpBlock(filters=128, kernel_size=4),
        UpBlock(filters=64, kernel_size=4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh',
    )

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
