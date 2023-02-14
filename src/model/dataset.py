import tensorflow as tf

def load_image(x_path, y_path):
    # read file from path
    x_file = tf.io.read_file(x_path)
    y_file = tf.io.read_file(y_path)

    # decode image
    x = tf.image.decode_png(x_file, channels=1)
    y = tf.image.decode_png(y_file, channels=1)

    # image shape is uneven, so we need to pad it to make it even
    x = tf.image.resize_with_crop_or_pad(x, 256, 256)
    y = tf.image.resize_with_crop_or_pad(y, 256, 256)

    # normalize image
    x = tf.cast(x, tf.float32) / 127.5 - 1
    y = tf.cast(y, tf.float32) / 127.5 - 1

    return x, y

def load_dataset(x_paths, y_paths, batch_size=8, shuffle=True):
    # create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))

    # load images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=32)

    # batch dataset
    dataset = dataset.batch(batch_size)

    # prefetch dataset
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

