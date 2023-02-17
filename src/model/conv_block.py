import tensorflow as tf
from src.model.norm import InstanceNormalization

# residual block for generator for down sampling
class DownBlock(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides, 
            padding, 
            activation, 
            use_bias, 
            kernel_initializer, 
            **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
        )
        self.instance_norm = InstanceNormalization()


    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.instance_norm(x)
        return x
    

# residual block for generator for up sampling
class UpBlock(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides, 
            padding, 
            activation, 
            use_bias, 
            kernel_initializer, 
            **kwargs):
        super(UpBlock, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            )

        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout(x)
        return x

