import tensorflow as tf

# residual block for generator for down sampling
class ResBlockDown(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides, 
            padding, 
            activation, 
            use_bias, 
            kernel_initializer, 
            kernel_regularizer, 
            **kwargs):
        super(ResBlockDown, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), 
            padding='same'
        )
        self.concat = tf.keras.layers.Concatenate()


    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.concat([x, inputs])
        x = self.pool(x)
        return x
    

# residual block for generator for up sampling
class ResBlockUp(tf.keras.layers.Layer):

    def __init__(
            self, 
            filters, 
            kernel_size, 
            strides, 
            padding, 
            activation, 
            use_bias, 
            kernel_initializer, 
            kernel_regularizer, 
            **kwargs):
        super(ResBlockUp, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            kernel_regularizer=kernel_regularizer
        )
        self.up = tf.keras.layers.UpSampling2D(
            size=(2, 2), 
            interpolation='bilinear'
        )
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.concat([x, inputs])
        x = self.up(x)
        return x

