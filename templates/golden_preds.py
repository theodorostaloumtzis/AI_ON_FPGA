import tensorflow as tf
import numpy as np

# QKeras imports
from qkeras import (
    QConv2D,
    QActivation,
    QDense,
    QBatchNormalization,
    QDepthwiseConv2D,
    QConv2DBatchnorm,  # fused conv + BN layer
    quantized_bits,
    quantized_relu,
)

# ---------------------------------------------------------------------------
# Register ALL custom QKeras objects so Keras can deserialize the model
# ---------------------------------------------------------------------------
QKerasCustomObjects = {
    "QConv2D": QConv2D,
    "QActivation": QActivation,
    "QDense": QDense,
    "QBatchNormalization": QBatchNormalization,
    "QDepthwiseConv2D": QDepthwiseConv2D,
    "QConv2DBatchnorm": QConv2DBatchnorm,
    "quantized_bits": quantized_bits,
    "quantized_relu": quantized_relu,
}

# ---------------------------------------------------------------------------
# Quantisation helper for ap_fixed<16,6>
# ---------------------------------------------------------------------------
def quantize_ap_fixed(x, total_bits: int = 16, int_bits: int = 6):
    """Clip + round *x* to signed ap_fixed<total_bits,int_bits>."""
    frac_bits = total_bits - int_bits
    scale = 2 ** frac_bits
    min_val = -2 ** int_bits
    max_val = (2 ** int_bits) - 1 / scale
    x = np.clip(x, min_val, max_val)
    return np.round(x * scale) / scale

# ---------------------------------------------------------------------------
# Load the saved quantised Keras model
# ---------------------------------------------------------------------------
with tf.keras.utils.custom_object_scope(QKerasCustomObjects):
    model = tf.keras.models.load_model("keras_model.keras", compile=False)

# ---------------------------------------------------------------------------
# Load full MNIST test set
# ---------------------------------------------------------------------------
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype(np.float32) / 255.0  # normalise to [0,1]
y_test = y_test

x_test = np.expand_dims(x_test, -1)           # shape: (10000, 28, 28, 1)
x_flat = x_test.reshape(x_test.shape[0], -1)  # flatten for export

# ---------------------------------------------------------------------------
# Inference + fixed-point quantisation
# ---------------------------------------------------------------------------
y_pred = model.predict(x_test)

# ---------------------------------------------------------------------------
# Save golden predictions
# ---------------------------------------------------------------------------

np.save("golden_preds.npy", y_pred)

print("Saved all 10,000 quantized inputs to 'golden_inputs.npy' and predictions to 'golden_preds.npy'")