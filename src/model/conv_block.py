import tensorflow as tf
from src.model.norm import InstanceNormalization

# residual block for generator for down sampling
class DownBlock(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            apply_norm=True,
            **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        initializer = tf.random_normal_initializer(0., 0.02)
        
        self.apply_norm = apply_norm

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
        )

        self.instance_norm = InstanceNormalization()

        self.leaky_relu = tf.keras.layers.LeakyReLU()


    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        if self.apply_norm:
            x = self.instance_norm(x, training=training)
        x = self.leaky_relu(x, training=training)
        return x
    

# residual block for generator for up sampling
class UpBlock(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            apply_dropout=False,
            is_tanh=False, 
            **kwargs):
        super(UpBlock, self).__init__(**kwargs)

        initializer = tf.random_normal_initializer(0., 0.02)

        self.apply_dropout = apply_dropout

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
        )
        
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

        if is_tanh:
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            self.activation = tf.keras.layers.ReLU()
        

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = self.activation(x, training=training)
        return x

