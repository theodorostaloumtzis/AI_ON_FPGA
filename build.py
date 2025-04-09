#!/usr/bin/env python
# coding: utf-8

import os
import time
import pprint
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# For typical data and ML
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, BatchNormalization, Activation, MaxPooling2D,
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1

# qkeras-related
from qkeras.autoqkeras import AutoQKeras, print_qmodel_summary
from qkeras import quantized_bits
from qkeras import QConv2D, QDense, QActivation


from qkeras import (
    QConv2D, QDense, QActivation, QBatchNormalization, QDepthwiseConv2D,
    quantized_bits, quantized_relu
)
from tensorflow.keras.utils import register_keras_serializable



# hls4ml-related
import hls4ml
# hls4ml's built-in plotting utility for printing configs
import utils.plotting as plotting
from hls4ml.report import read_vivado_report

# Utility modules (you must ensure these are available)
from utils.model_utils import save_model

# Adjust environment variables to match your local paths
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2024.2:/tools/Xilinx/Vitis/2020.1/'
os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
os.environ['PATH'] = '/tools/Xilinx/Vitis_HLS/2024.2/bin:' + os.environ['PATH']
os.environ['PATH'] = '/tools/Xilinx/Vitis/2020.1/bin:' + os.environ['PATH']

###############################################################################
# Basic data loading and model-building
###############################################################################


def prepare_data():
    """
    Load and preprocess the MNIST dataset, then return train/val/test as tf.data.Datasets.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Reshape to add channel dimension (28x28x1)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test  = x_test.reshape((-1, 28, 28, 1))

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    # Split off a validation set
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # Create tf.data.Dataset objects
    batch_size = 1024
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                                .shuffle(10000) \
                                .batch(batch_size)
    val_data   = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                                .batch(batch_size)
    test_data  = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
                                .batch(batch_size)

    return train_data, val_data, test_data


def build_model(input_shape=(28, 28, 1), n_classes=10):
    """
    Build and compile a baseline CNN model for MNIST.
    Returns the compiled keras Model.
    """
    filters_per_conv_layer = [16, 8]
    neurons_per_dense_layer = [24]

    x_in = Input(input_shape)
    x = x_in

    # Convolutional blocks
    for i, f in enumerate(filters_per_conv_layer):
        print(f"Adding convolutional block {i} with N={f} filters")
        x = Conv2D(
            f,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=False,
            name='conv_{}'.format(i),
        )(x)
        x = BatchNormalization(name='bn_conv_{}'.format(i))(x)
        x = Activation('relu', name='conv_act_%i' % i)(x)
        x = MaxPooling2D(pool_size=(2,2), name='pool_{}'.format(i))(x)

    # Flatten + Dense blocks
    x = Flatten()(x)
    for i, n in enumerate(neurons_per_dense_layer):
        print(f"Adding dense block {i} with N={n} neurons")
        x = Dense(
            n,
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=False,
            name='dense_%i' % i
        )(x)
        x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
        x = Activation('relu', name='dense_act_%i' % i)(x)

    # Final classification
    x = Dense(n_classes, name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='keras_baseline')

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=3e-3, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=True
    )

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    return model


def train_model(model, train_data, val_data, test_data, n_epochs=10):
    """
    Train the given model on train_data and val_data, evaluate on test_data.
    Returns the trained model.
    """
    print("\n--- Training Baseline Keras Model ---\n")
    model.fit(train_data, epochs=n_epochs, validation_data=val_data, verbose=2)

    # Evaluate
    loss, accuracy = model.evaluate(test_data, verbose=2)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Print QKeras summary if available
    try:
        print("\nQKeras model summary:\n")
        print_qmodel_summary(model)
    except:
        pass

    return model

###############################################################################
# AutoQKeras-related logic
###############################################################################


def run_autoqkeras_tuning(model, train_data, val_data, n_epochs=10, max_trials=5):
    """
    Illustrates how to run AutoQKeras for quantization tuning.
    It uses the same 'epochs' as the baseline training, and
    it accepts 'max_trials' for the Bayesian or random search.

    Returns the AutoQKeras object (you can retrieve the best model with .get_best_model()).
    """
    # Example quantization config
    quantization_config = {
        "kernel": {
            "quantized_bits(2,0,1,alpha=1.0)": 2,
            "quantized_bits(4,0,1,alpha=1.0)": 4,
            "quantized_bits(6,0,1,alpha=1.0)": 6,
            "quantized_bits(8,0,1,alpha=1.0)": 8,
        },
        "bias": {
            "quantized_bits(2,0,1,alpha=1.0)": 2,
            "quantized_bits(4,0,1,alpha=1.0)": 4,
            "quantized_bits(6,0,1,alpha=1.0)": 6,
            "quantized_bits(8,0,1,alpha=1.0)": 8,
        },
        "activation": {
            "quantized_relu(3,1)": 3,
            "quantized_relu(4,2)": 4,
            "quantized_relu(8,2)": 8,
            "quantized_relu(8,4)": 8,
            "quantized_relu(16,6)": 16,
        },
        "linear": {
            "quantized_bits(2,0,1,alpha=1.0)": 2,
            "quantized_bits(4,0,1,alpha=1.0)": 4,
            "quantized_bits(6,0,1,alpha=1.0)": 6,
            "quantized_bits(8,0,1,alpha=1.0)": 8,
        },
    }

    limit = {
        "conv": [8, 16],
        "dense": [8, 16],
        "act": [16]
    }

    # Example 'goal' using energy
    goal_energy = {
        "type": "energy",
        "params": {
            "delta_p": 8.0,
            "delta_n": 8.0,
            "rate": 2.0,
            "stress": 1.0,
            "process": "horowitz",
            "parameters_on_memory": ["sram", "sram"],
            "activations_on_memory": ["sram", "sram"],
            "rd_wr_on_io": [False, False],
            "min_sram_size": [0, 0],
            "source_quantizers": ["fp32"],
            "reference_internal": "int8",
            "reference_accumulator": "int32",
        },
    }

    run_config = {
        "goal": goal_energy,
        "quantization_config": quantization_config,
        "learning_rate_optimizer": False,
        "transfer_weights": False,
        "mode": "bayesian",  # or "random", "hyperband" etc.
        "seed": 42,
        "limit": limit,
        "tune_filters": "none",
        "tune_filters_exceptions": "",
        "distribution_strategy": None,
        "max_trials": max_trials,  # <-- Using your custom flag
    }

    autoqk = AutoQKeras(
      model=model,
      output_dir="autoqk_results",
      **run_config
    )


    space = autoqk.tuner.oracle.get_space()
    print("\nRegistered hyperparameters in AutoQKeras:")
    for hp in space.space:
        print(f" • {hp.name}: {hp.values}")

    # We'll reuse the same epochs used for the baseline.
    print(f"\n--- Running AutoQKeras search for {max_trials} trials, {n_epochs} epochs each ---\n")
    autoqk.fit(
        x=train_data,
        validation_data=val_data,
        epochs=n_epochs  # each trial trains for these many epochs
    )
    return autoqk

###############################################################################
# HLS conversion, bitstream, etc.
###############################################################################


def evaluate_model(model, test_data, do_bitstream=False, board_name="ZCU104", part= 'xc7z020clg400-1'):
    """
    Evaluate the model, save it, and convert to HLS.
    If do_bitstream=True, we use the 'VivadoAccelerator' backend with a board name.
    Otherwise we use a normal backend (e.g. 'Vitis') for typical HLS flows.

    Returns (hls_model_aq, output_directory).
    """
    print("\n--- Evaluating/Saving Model ---\n")

    # Save the model
    model_dir = "models"
    model_name = "keras_baseline"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    save_model(model, model_path)
    print(f"Model saved to {model_path}.h5")

    # Choose the correct backend and board config
    if do_bitstream:
        print("\nUsing VivadoAccelerator backend for bitstream generation.")
        backend = "VivadoAccelerator"
    else:
        # For normal synthesis, you can use "Vitis" or "Vivado"
        print("\nUsing Vitis backend for normal flow.")
        backend = "Vitis"

    # Convert to HLS
    print("\n--- Converting to HLS with hls4ml ---\n")
    hls_config_aq = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_config_aq['Model']['ReuseFactor'] = 8
    hls_config_aq['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config_aq['LayerName']['output_softmax']['Strategy'] = 'Stable'
    plotting.print_dict(hls_config_aq)

    save_path = os.path.join("Projects", "AutoQKeras")
    os.makedirs(save_path, exist_ok=True)

    cfg_aq = hls4ml.converters.create_config(backend=backend)
    cfg_aq['IOType'] = 'io_stream'  # Must set this if using CNNs!
    cfg_aq['HLSConfig'] = hls_config_aq
    cfg_aq['KerasModel'] = model
    cfg_aq['OutputDir'] = save_path

    # For bitstream generation with VivadoAccelerator, must specify Board
    if do_bitstream:
        cfg_aq['Board'] = board_name
    else:
        # Possibly specify a Xilinx part if you're doing normal Vitis or Vivado flow
        cfg_aq['XilinxPart'] = part

    hls_model_aq = hls4ml.converters.keras_to_hls(cfg_aq)
    hls_model_aq.compile()

    print("hls4ml model compilation complete.")
    return hls_model_aq, save_path


def finalize_hls_project(hls_model, project_dir, do_synth=False, do_report=False, do_bitstream=False):
    """
    We allow EITHER:
      --synth (Vivado HLS-based, csim=False, synth=True, vsynth=True), OR
      --bitstream (VivadoAccelerator-based, csim=False, export=True, bitfile=True)

    If do_report=True, we read & display the Vivado resource/timing after building.

    By design, main() disallows both do_synth AND do_bitstream at once.
    """
    if do_synth:
        print("\n--- Running HLS Synthesis build (Vivado HLS) ---\n")
        hls_model.build(csim=False, synth=True, vsynth=True)
        print("Synthesis flow done.")
        if do_report:
            print("\n--- Reading Vivado Report (Synthesis) ---\n")
            vivado_report = read_vivado_report(project_dir)
            pprint.pprint(vivado_report)

    elif do_bitstream:
        print("\n--- Running HLS Build for Bitstream (Vivado Accelerator) ---\n")
        hls_model.build(csim=False, export=True, bitfile=True)
        print("Bitstream generation build done.")
        if do_report:
            print("\n--- Reading Vivado Report (Bitstream Build) ---\n")
            vivado_report = read_vivado_report(project_dir)
            pprint.pprint(vivado_report)

    else:
        print("\nNo synthesis or bitstream build requested. Done.")


def process_best_autoqkeras_model(best_model, train_data, val_data, test_data, n_epochs):
    """
    Fine-tune, recompile, evaluate, and prepare the best AutoQKeras model for HLS.
    Returns the finalized model.
    """
    print("\n--- Processing Best AutoQKeras Model ---\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]

    start = time.time()
    history = best_model.fit(train_data, epochs=n_epochs, validation_data=val_data, callbacks=callbacks, verbose=1)
    end = time.time()
    print('\n Training completed in {:.2f} minutes\n'.format((end - start) / 60.0))

    best_model.save_weights("autoqkeras_cnn_weights.h5")

    # Rebuild and recompile model
    layers = [l for l in best_model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        x = layers[i](x)

    new_model = Model(inputs=[layers[0].input], outputs=[x])
    new_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
        metrics=["accuracy"]
    )
    new_model.summary()
    new_model.load_weights("autoqkeras_cnn_weights.h5")

    # Evaluation
    results = new_model.evaluate(test_data, verbose=2)
    metrics = dict(zip(new_model.metrics_names, results))
    print("\nAutoQKeras best model test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return new_model


###############################################################################
# Main entry point
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, optionally run AutoQKeras, and optionally do HLS synthesis or bitstream generation.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs for both baseline model and AutoQKeras search.")
    parser.add_argument("--synth", action="store_true", help="Run HLS synthesis with Vivado HLS.")
    parser.add_argument("--report", action="store_true", help="Read Vivado utilization report after building.")
    parser.add_argument("--bitstream", action="store_true", help="Run bitstream generation with Vivado Accelerator.")
    parser.add_argument("--board", type=str, default="ZCU104", help="Board name for bitstream generation if --bitstream is set.")
    parser.add_argument("--autoqk", action="store_true", help="Whether to run AutoQKeras search after the baseline training.")
    parser.add_argument("--max-trials", type=int, default=5, help="Max trials for the AutoQKeras search.")
    args = parser.parse_args()

    if args.synth and args.bitstream:
        print("\nERROR: --synth and --bitstream cannot both be used in the same run.")
        return

    train_data, val_data, test_data = prepare_data()
    model = build_model(input_shape=(28, 28, 1), n_classes=10)
    model = train_model(model, train_data, val_data, test_data, n_epochs=args.epochs)

    if args.autoqk:
        print("\n--- Running AutoQKeras Search ---\n")
        autoqk = run_autoqkeras_tuning(model, train_data, val_data,
                                       n_epochs=args.epochs,
                                       max_trials=args.max_trials)

        best_model = autoqk.get_best_model()
        model = process_best_autoqkeras_model(best_model, train_data, val_data, test_data, args.epochs)

    hls_model, hls_project_path = evaluate_model(model, test_data,
                                                 do_bitstream=args.bitstream,
                                                 board_name=args.board)

    finalize_hls_project(
        hls_model=hls_model,
        project_dir=hls_project_path,
        do_synth=args.synth,
        do_report=args.report,
        do_bitstream=args.bitstream
    )



if __name__ == "__main__":
    main()
