# data_pipeline.py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Load and preprocess the MNIST dataset, then return train/val/test as tf.data.Datasets.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test  = x_test.reshape((-1, 28, 28, 1))

    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    batch_size = 1024
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    val_data   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_data  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_data, val_data, test_data