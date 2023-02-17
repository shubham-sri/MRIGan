import tensorflow as tf
from src.model.res_block import DownBlock, UpBlock

# U-Net generator using residual blocks for down and up sampling
class Generator(tf.keras.Model):

    def __init__(self,**kwargs):
        super(Generator, self).__init__(**kwargs)

        # down sampling
        self.down1 = DownBlock(
            filters=32, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            padding='same', 
            activation=tf.keras.layers.LeakyReLU(), 
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02), 
            
        )
        self.down2 = DownBlock(
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )
        self.down3 = DownBlock(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )
        self.down4 = DownBlock(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )
        self.down5 = DownBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )
        self.down6 = DownBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )
        self.down7 = DownBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )

        # bottleneck
        self.bottleneck = DownBlock(
            filters=256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            
        )

        # up sampling
        self.up1 = UpBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up2 = UpBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up3 = UpBlock(
            filters=128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up4 = UpBlock(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up5 = UpBlock(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up6 = UpBlock(
            filters=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        self.up7 = UpBlock(
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(),
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )
        

        # conv
        self.conv = UpBlock(
            filters=1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            activation='tanh',
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
        )

        # concat
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x = inputs

        # down sampling
        x1 = self.down1(x, training=training)
        x2 = self.down2(x1, training=training)
        x3 = self.down3(x2, training=training)
        x4 = self.down4(x3, training=training)
        x5 = self.down5(x4, training=training)
        x6 = self.down6(x5, training=training)
        x7 = self.down7(x6, training=training)

        # bottleneck
        x = self.bottleneck(x7, training=training)

        # up sampling
        x = self.up1(x, training=training)
        x = self.concat([x, x7])
        x = self.up2(x, training=training)
        x = self.concat([x, x6])
        x = self.up3(x, training=training)
        x = self.concat([x, x5])
        x = self.up4(x, training=training)
        x = self.concat([x, x4])
        x = self.up5(x, training=training)
        x = self.concat([x, x3])
        x = self.up6(x, training=training)
        x = self.concat([x, x2])
        x = self.up7(x, training=training)
        x = self.concat([x, x1])

        # conv
        x = self.conv(x)

        return x

