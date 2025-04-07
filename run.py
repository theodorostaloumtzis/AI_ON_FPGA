import os
import time
import pprint
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Conv2D, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qkeras import *
from qkeras.autoqkeras import AutoQKeras
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
from qkeras.utils import print_qmodel_summary

import hls4ml
import plotting


# Set up environment variables
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2024.2'
os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']


def load_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return train_test_split(x_train, y_train, test_size=0.1, random_state=42), x_test, y_test


def build_model(input_shape=(28, 28, 1), n_classes=10):
    filters_per_conv_layer = [16, 16, 24]
    neurons_per_dense_layer = [42, 64]
    x = x_in = Input(input_shape)

    for i, f in enumerate(filters_per_conv_layer):
        x = Conv2D(f, kernel_size=3, strides=1, kernel_initializer='lecun_uniform',
                   kernel_regularizer=l1(0.0001), use_bias=False, name=f'conv_{i}')(x)
        x = BatchNormalization(name=f'bn_conv_{i}')(x)
        x = Activation('relu', name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size=2, name=f'pool_{i}')(x)

    x = Flatten()(x)

    for i, n in enumerate(neurons_per_dense_layer):
        x = Dense(n, kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),
                  name=f'dense_{i}', use_bias=False)(x)
        x = BatchNormalization(name=f'bn_dense_{i}')(x)
        x = Activation('relu', name=f'dense_act_{i}')(x)

    x = Dense(n_classes, name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)

    return Model(inputs=x_in, outputs=x_out, name='keras_baseline')


def compile_model(model):
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


def run_energy_analysis(model):
    q = run_qtools.QTools(model, process="horowitz", source_quantizers=[quantized_bits(16, 5, 1)],
                          is_inference=True, weights_path=None, keras_quantizer="fp16",
                          keras_accumulator="fp16", for_reference=False)
    q.qtools_stats_print()
    energy_dict = q.pe(weights_on_memory="fixed", activations_on_memory="fixed",
                       min_sram_size=8 * 16 * 1024 * 1024, rd_wr_on_io=False)
    profile = q.extract_energy_profile(qtools_settings.cfg.include_energy, energy_dict)
    total_energy = q.extract_energy_sum(qtools_settings.cfg.include_energy, energy_dict)
    pprint.pprint(profile)
    print("Total energy: {:.6f} uJ".format(total_energy / 1e6))


def run_autoqkeras(model, train_data, val_data):
    quant_config = {
        "kernel": {f"quantized_bits({b},0,1,alpha=1.0)": b for b in [2, 4, 6, 8]},
        "bias": {f"quantized_bits({b},0,1,alpha=1.0)": b for b in [2, 4, 6, 8]},
        "activation": {
            "quantized_relu(3,1)": 3,
            "quantized_relu(4,2)": 4,
            "quantized_relu(8,2)": 8,
            "quantized_relu(8,4)": 8,
            "quantized_relu(16,6)": 16,
        },
        "linear": {f"quantized_bits({b},0,1,alpha=1.0)": b for b in [2, 4, 6, 8]},
    }

    goal = {
        "type": "energy",
        "params": {
            "delta_p": 8.0, "delta_n": 8.0, "rate": 2.0, "stress": 1.0,
            "process": "horowitz",
            "parameters_on_memory": ["sram", "sram"],
            "activations_on_memory": ["sram", "sram"],
            "rd_wr_on_io": [False, False],
            "min_sram_size": [0, 0],
            "source_quantizers": ["fp32"],
            "reference_internal": "int8",
            "reference_accumulator": "int32",
        }
    }

    config = {
        "goal": goal,
        "quantization_config": quant_config,
        "learning_rate_optimizer": False,
        "transfer_weights": False,
        "mode": "bayesian",
        "seed": 42,
        "limit": {"conv": [8, 8, 16], "dense": [8, 8, 16], "act": [16]},
        "tune_filters": "layer",
        "tune_filters_exceptions": "^output",
        "distribution_strategy": None,
        "max_trials": 5,
    }

    autoqk = AutoQKeras(model=model, output_dir="autoqk_results", **config)
    autoqk.fit(x=train_data, validation_data=val_data, epochs=15)
    return autoqk.get_best_model()


def train_final_model(model, train_data, val_data, n_epochs=10):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]
    start = time.time()
    history = model.fit(train_data, epochs=n_epochs, validation_data=val_data, callbacks=callbacks, verbose=1)
    print('\n It took {:.2f} minutes to train!\n'.format((time.time() - start) / 60.0))
    return history


def export_hls_model(model, x_test, y_test):
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_config['Model']['ReuseFactor'] = 8
    hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
    plotting.print_dict(hls_config)

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType'] = 'io_stream'
    cfg['HLSConfig'] = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir'] = 'autoqkeras_cnn/'
    cfg['XilinxPart'] = 'xczu5ev-sfvc784-1-i'

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()

    acc_keras = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
    acc_hls4ml = accuracy_score(np.argmax(y_test, axis=1), np.argmax(hls_model.predict(np.ascontiguousarray(x_test)), axis=1))
    print("Accuracy AutoQ Keras:  {}".format(acc_keras))
    print("Accuracy AutoQ hls4ml: {}".format(acc_hls4ml))

    hls_model.build(csim=False, synth=True, vsynth=True)


def main():
    (x_train, y_train, x_val, y_val), x_test, y_test = load_preprocess_data()
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(1024)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1024)
    
    model = build_model()
    model = compile_model(model)
    
    run_energy_analysis(model)

    best_model = run_autoqkeras(model, train_data, val_data)
    print_qmodel_summary(best_model)

    best_model.save_weights("autoqkeras_cnn_weights.h5")
    best_model.load_weights("autoqkeras_cnn_weights.h5")

    train_final_model(best_model, train_data, val_data)
    export_hls_model(best_model, x_test, y_test)


if __name__ == "__main__":
    main()
