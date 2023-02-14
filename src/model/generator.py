import tensorflow as tf
from src.model.res_block import ResBlockDown, ResBlockUp

# U-Net generator using residual blocks for down and up sampling
class Generator(tf.keras.Model):

    def __init__(self,**kwargs):
        super(Generator, self).__init__(**kwargs)

        # down sampling
        self.down1 = ResBlockDown(
            filters=64, 
            kernel_size=(5, 5), 
            strides=(1, 1), 
            padding='same', 
            activation=tf.keras.layers.LeakyReLU(alpha=0.2), 
            use_bias=False,
            kernel_initializer='he_normal', 
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down2 = ResBlockDown(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down3 = ResBlockDown(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down4 = ResBlockDown(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down5 = ResBlockDown(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down6 = ResBlockDown(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.down7 = ResBlockDown(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )

        # bottleneck
        self.bottleneck = ResBlockDown(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )

        # up sampling
        self.up1 = ResBlockUp(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up2 = ResBlockUp(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up3 = ResBlockUp(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up4 = ResBlockUp(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up5 = ResBlockUp(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU(),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up6 = ResBlockUp(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        self.up7 = ResBlockUp(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )
        

        # conv
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
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

        print(x.shape)

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

