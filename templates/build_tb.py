import tensorflow as tf
import numpy as np
import os

#  QKeras imports
from qkeras import (
    QConv2D,
    QActivation,
    QDense,
    QBatchNormalization,
    QDepthwiseConv2D,
    QConv2DBatchnorm,  # <─ new: fused conv + BN layer
    quantized_bits,
    quantized_relu,
)

# --- Register ALL custom QKeras objects ------------------------------------
#   Keras needs these classes & quantizers at deserialisation time.
#   Add any that your model might contain so load_model() doesn’t break.
QKerasCustomObjects = {
    "QConv2D": QConv2D,
    "QActivation": QActivation,
    "QDense": QDense,
    "QBatchNormalization": QBatchNormalization,
    "QDepthwiseConv2D": QDepthwiseConv2D,
    "QConv2DBatchnorm": QConv2DBatchnorm,  # <─ new entry fixes the error
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
# Load the saved quantised Keras model (compiled=False → no custom losses needed)
# ---------------------------------------------------------------------------

with tf.keras.utils.custom_object_scope(QKerasCustomObjects):
    model = tf.keras.models.load_model("keras_model.keras", compile=False)

# ---------------------------------------------------------------------------
# Prepare a (shuffled) subset of MNIST for test‑bench generation
# ---------------------------------------------------------------------------

(x_test, y_test), _ = tf.keras.datasets.mnist.load_data()

N_SAMPLES = 100
np.random.seed(42)
indices = np.random.permutation(len(x_test))[:N_SAMPLES]

x_test = x_test[indices].astype(np.float32) / 255.0  # normalise to [0,1]
y_test = y_test[indices]

x_test = np.expand_dims(x_test, -1)           # shape: (N, 28, 28, 1)
x_flat = x_test.reshape(N_SAMPLES, -1)        # flatten for .dat export

# ---------------------------------------------------------------------------
# Inference + fixed‑point quantisation
# ---------------------------------------------------------------------------

y_pred = model.predict(x_test, verbose=0)

x_quantized = quantize_ap_fixed(x_flat)
y_quantized = quantize_ap_fixed(y_pred)

# ---------------------------------------------------------------------------
# Dump test‑bench stimuli and golden outputs
# ---------------------------------------------------------------------------

os.makedirs("tb_data", exist_ok=True)
np.savetxt("tb_data/tb_input_features.dat", x_quantized, fmt="%.6f")
np.savetxt("tb_data/tb_output_predictions.dat", y_quantized, fmt="%.6f")

print("✅ Quantized data written to 'tb_data/tb_input_features.dat' and 'tb_data/tb_output_predictions.dat'")
