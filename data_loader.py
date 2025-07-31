import tensorflow as tf

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    # Improve performance using caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
