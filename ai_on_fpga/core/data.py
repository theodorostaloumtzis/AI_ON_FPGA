"""
Minimal MNIST data pipeline (replace with your dataset loader if needed).
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def prepare(cfg: dict):
    batch = cfg.get("batch", 128)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., None] / 255.0
    x_test  = x_test [..., None] / 255.0

    ds_train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(batch)
    )
    ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch)
    return ds_train, ds_val
