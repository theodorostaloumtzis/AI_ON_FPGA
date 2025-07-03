import os
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# Enable TensorFlow 2.x behavior explicitly
tf.enable_v2_behavior()

# --- Xilinx Toolchain Setup ---
VITIS_PATH = '/tools/Xilinx/Vitis/2024.2'
VIVADO_PATH = '/tools/Xilinx/Vivado/2024.2/bin'
VITIS_HLS_PATH = '/tools/Xilinx/Vitis_HLS/2024.2/bin'

# Set environment variables for Vitis
os.environ['XILINX_VITIS'] = VITIS_PATH
os.environ['PATH'] = f"{VIVADO_PATH}:{VITIS_HLS_PATH}:" + os.environ['PATH']

# --- Optional sanity checks ---
for path in [VITIS_PATH, VIVADO_PATH, VITIS_HLS_PATH]:
    if not os.path.exists(path):
        print(f"⚠️  Warning: {path} does not exist!")

print("✅ Environment configured for Xilinx toolchain.")


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten the images for MLP: 28x28 → 784
x_train = x_train.reshape((-1, 784))
x_test  = x_test.reshape((-1, 784))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Split off a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Create tf.data.Dataset objects
batch_size = 1024

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
val_data   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_data  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Metadata
n_epochs = 10
train_size = len(x_train)
input_shape = (784,)  # flat input for MLP
n_classes = 10


from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Input(shape=input_shape),       # (784,)
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=n_epochs,
    verbose=1
)

max_layer_size = 0
for layer in model.layers:
    if layer.__class__.__name__ in ['Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
        if layersize > 4096:  # assuming that shape[0] is batch, i.e., 'None'
            if layersize > max_layer_size:
                max_layer_size = layersize
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name, layersize))

print(f"The max_layer_size is {max_layer_size}")


import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

NSTEPS = int(train_size * 0.9) // batch_size
print('Number of training steps per epoch is {}'.format(NSTEPS))
# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.50, begin_step=NSTEPS * 2, end_step=NSTEPS * 10, frequency=NSTEPS
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

model_pruned = tf.keras.models.clone_model(model, clone_function=pruneFunction)

models_path = 'models'

train = True  # True if you want to retrain, false if you want to load a previsously trained model

n_epochs = 30

save_path = os.path.join(models_path, 'pruned_cnn_model.h5')

if train:
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)

    model_pruned.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        pruning_callbacks.UpdatePruningStep(),
    ]

    start = time.time()
    model_pruned.fit(train_data, epochs=n_epochs, validation_data=val_data, callbacks=callbacks)
    end = time.time()

    print('It took {} minutes to train Keras model'.format((end - start) / 60.0))
    model_pruned.save(save_path)


else:
    from qkeras.utils import _add_supported_quantized_objects
    from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

    co = {}
    _add_supported_quantized_objects(co)
    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model_pruned = tf.keras.models.load_model('pruned_cnn_model.h5', custom_objects=co)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation
from qkeras import QDense, QActivation
from tensorflow.keras.regularizers import l1

# Example MLP configuration
input_shape = (784,)  # for MNIST flattened
neurons_per_dense_layer = [64, 32]  # feel free to change
n_classes = 10

x_in = Input(shape=input_shape, name='input_layer')
x = x_in

# Add quantized Dense + BN + Activation blocks
for i, n in enumerate(neurons_per_dense_layer):
    print(f"Adding QDense block {i} with N={n} neurons")
    x = QDense(
        n,
        kernel_quantizer="quantized_bits(6, 0, alpha=1)",
        bias_quantizer="quantized_bits(6, 0, alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name=f'dense_{i}',
        use_bias=True
    )(x)
    x = BatchNormalization(name=f'bn_dense_{i}')(x)
    x = QActivation('quantized_relu(6)', name=f'dense_act_{i}')(x)

# Output layer (not quantized in this case — can be made quantized too)
x = QDense(n_classes, name='output_dense')(x)
x_out = Activation('softmax', name='output_softmax')(x)

# Build model
qmodel = Model(inputs=x_in, outputs=x_out, name='qkeras_mlp')


train = True

q_save_path = os.path.join(models_path, 'quantized_pruned_cnn_model.h5')

n_epochs = 30
if train:
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    qmodel_pruned.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        pruning_callbacks.UpdatePruningStep(),
    ]

    start = time.time()
    history = qmodel_pruned.fit(train_data, epochs=n_epochs, validation_data=val_data, callbacks=callbacks, verbose=1)
    end = time.time()
    print('\n It took {} minutes to train!\n'.format((end - start) / 60.0))

    qmodel_pruned.save(q_save_path)

else:
    from qkeras.utils import _add_supported_quantized_objects
    from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

    co = {}
    _add_supported_quantized_objects(co)
    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    qmodel_pruned = tf.keras.models.load_model('quantized_pruned_cnn_model.h5', custom_objects=co)
    
    
predict_baseline = model_pruned.predict(x_test)
test_score_baseline = model_pruned.evaluate(x_test, y_test)

predict_qkeras = qmodel_pruned.predict(x_test)
test_score_qkeras = qmodel_pruned.evaluate(x_test, y_test)

print(f'Keras accuracy = {test_score_baseline[1]*100:.2f}% , QKeras 6-bit accuracy = {test_score_qkeras[1]*100:.2f}%')

from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

model = tf.keras.models.load_model(save_path, custom_objects=co)
model = strip_pruning(model)

qmodel = tf.keras.models.load_model(q_save_path, custom_objects=co)
qmodel = strip_pruning(qmodel)
project_folder = 'Projects'


import hls4ml

# Auto-generate base config
hls_config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name',
    backend='Vitis',
    default_precision='ap_fixed<16,6>'
)

# Custom performance overrides
for layer_name, layer_cfg in hls_config['LayerName'].items():
    layer_cfg['Strategy'] = 'Latency'
    layer_cfg['ReuseFactor'] = 32
    if 'FifoDepth' not in layer_cfg:
        layer_cfg['FifoDepth'] = 4


save_proj_path = os.path.join(project_folder, 'Baseline')

# Convert and compile
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=hls_config,
    backend='Vitis',
    output_dir=save_proj_path,
    part='xqzu5ev-sfrc784-1L-i',
    io_type='io_stream',
    clock_period=5,
)
hls_model.compile()

# Generate config from QKeras model
hls_config_q = hls4ml.utils.config_from_keras_model(
    qmodel,
    granularity='name',
    backend='Vitis',
)

# Inject optimizations
hls_config_q['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config_q['Model']['PruneReuseFactorStrategy'] = 'load_balance'

for lname, lcfg in hls_config_q['LayerName'].items():
    lcfg['Strategy'] = 'Latency'
    lcfg['ReuseFactor'] = 32
    


save_proj_path = os.path.join(project_folder, 'Quantized')

# Convert and compile
hls_model_q = hls4ml.converters.convert_from_keras_model(
    qmodel,
    hls_config=hls_config_q,
    output_dir=save_proj_path,
    backend='Vitis',
    io_type='io_stream',
    clock_period=5,
    part = 'xqzu5ev-sfrc784-1L-i',
)

hls_model_q.compile()

y_predict = model.predict(x_test)
y_predict_hls4ml = hls_model.predict(np.ascontiguousarray(x_test))

y_predict_q = qmodel.predict(x_test)
y_predict_hls4ml_q = hls_model_q.predict(np.ascontiguousarray(x_test))

synth = True  # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
if synth:
    hls_model_q.build(csim=False, synth=True, export=True)

synth = True  # Only if you want to synthesize the models yourself (>1h per model) rather than look at the provided reports.
if synth:
    hls_model.build(csim=False, synth=True, export=True)
    