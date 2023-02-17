import tensorflow as tf

def load_image(x_path):
    # read file from path
    x_file = tf.io.read_file(x_path)

    # decode image
    x = tf.image.decode_png(x_file, channels=1)

    # image shape is uneven, so we need to pad it to make it even
    x = tf.image.resize_with_crop_or_pad(x, 256, 256)

    # random flip
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    # normalize image
    x = tf.cast(x, tf.float32) / 127.5 - 1

    return x

def single_dataset(x_paths, batch_size=8, shuffle=True):
    # create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices(x_paths)

    # load images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2)

    # batch dataset
    dataset = dataset.batch(batch_size)

    # prefetch dataset
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_dataset(x_paths, y_paths, batch_size=8, shuffle=True):
    # create dataset from paths

    x_dataset = single_dataset(x_paths, batch_size, shuffle)
    y_dataset = single_dataset(y_paths, batch_size, shuffle)

    # zip datasets
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset
