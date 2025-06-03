import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax

def get_dataset(dataset_name, batch_size, is_train, max_sequence_length=None, debug_overfit=False):
    tf.random.set_seed(42 + jax.process_index())
    print("Loading dataset")
    if 'imagenet256' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            if is_train:
                image = tf.image.random_flip_left_right(image)
                if 'augment' in dataset_name:
                    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
                        tf.shape(image),
                        tf.zeros([0, 0, 4], tf.float32),
                        area_range=(5 / 100, 100 / 100),
                        aspect_ratio_range=(0.75, 1.33),
                        min_object_covered=0,  # Don't enforce a minimum area.
                        use_image_if_no_bounding_boxes=True
                    )
                    offset_y, offset_x, _ = tf.unstack(begin)
                    target_height, target_width, _ = tf.unstack(crop_size)
                    image = tf.slice(image, [offset_y, offset_x, 0], [target_height, target_width, 3])
                    image = tf.image.resize(image, (256, 256), antialias=True, method=tf.image.ResizeMethod.BILINEAR)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, data['label']

        split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        dataset = tfds.load('imagenet2012', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'celebahq256':
        def deserialization_fn(data):
            image = data['image']
            image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']

        split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        dataset = tfds.load('celebahq256', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.shuffle(20000, seed=42+jax.process_index(), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'lsunchurch':
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, 0 # No label

        split = tfds.split_for_jax_process('church-train' if is_train else 'church-test', drop_remainder=True)
        dataset = tfds.load('lsunc', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'openwebtext':
        # This does a naive tokenization scheme. It tokenizes, then cuts off at sequence_length. Otherwise it pads.
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        def deserialization_fn(data):
            text = data['text']
            return text

        split = tfds.split_for_jax_process('train[:95%]' if is_train else 'train[95%:]', drop_remainder=True)
        dataset = tfds.load('openwebtext', split=split)
        if debug_overfit:
            dataset = dataset.take(16)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)

        def batch_iterator():
            while True:
                # The +1 is to account for inputs/targets being offset.
                tokens = np.ones((batch_size, max_sequence_length+1), dtype=np.uint32) * enc.eot_token
                for b in range(batch_size):
                    token_list = []
                    while len(token_list) < max_sequence_length+1:
                        text = next(dataset)
                        token_list += enc.encode_ordinary(text[0].decode("utf-8")) + [enc.eot_token]
                    tokens[b] = np.array(token_list[:max_sequence_length+1], dtype=np.uint32)
                yield tokens, None

        return batch_iterator()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")