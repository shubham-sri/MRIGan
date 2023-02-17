import tensorflow as tf
from src.model.conv_block import DownBlock

def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    x = inputs
    x = DownBlock(filters=64, kernel_size=4, apply_norm=False)(x)
    x = DownBlock(filters=128, kernel_size=4)(x)
    x = DownBlock(filters=256, kernel_size=4)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = DownBlock(filters=512, kernel_size=4)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False,
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=x)