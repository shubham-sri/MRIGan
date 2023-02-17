import tensorflow as tf
from model.conv_block import DownBlock

class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        # down sampling
        
        self.down1 = DownBlock(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down2 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down3 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down4 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down5 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down6 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down7 = DownBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        self.down8 = DownBlock(
            filters=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

    def call(self, inputs, training=None):
        x = self.down1(inputs, training=training)
        x = self.down2(x, training=training)
        x = self.down3(x, training=training)
        x = self.down4(x, training=training)
        x = self.down5(x, training=training)
        x = self.down6(x, training=training)
        x = self.down7(x, training=training)
        x = self.down8(x, training=training)
        return x