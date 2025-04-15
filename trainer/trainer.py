# trainer.py
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QDense, QConv2D, QActivation, QBatchNormalization
from qkeras import quantized_bits, quantized_relu
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Conv2D, Activation, BatchNormalization
from qkeras.autoqkeras import print_qmodel_summary

def train_model(model, train_data, val_data, test_data, n_epochs=10):
    print("\n--- Training Baseline Keras Model ---\n")
    model.fit(train_data, epochs=n_epochs, validation_data=val_data, verbose=2)
    loss, accuracy = model.evaluate(test_data, verbose=2)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    try:
        print("\nQKeras model summary:\n")
        print_qmodel_summary(model)
    except:
        pass

    return model

def quantize_model(model):
    def convert_layer(layer):
        cfg = layer.get_config()

        if isinstance(layer, Dense):
            return QDense(
                units=cfg["units"],
                name=cfg["name"],
                kernel_initializer=cfg["kernel_initializer"],
                kernel_regularizer=cfg["kernel_regularizer"],
                use_bias=cfg["use_bias"],
                kernel_quantizer="quantized_bits(8,0,1)",
                bias_quantizer="quantized_bits(8,0,1)"
            )

        elif isinstance(layer, Conv2D):
            return QConv2D(
                filters=cfg["filters"],
                kernel_size=cfg["kernel_size"],
                strides=cfg["strides"],
                padding=cfg["padding"],
                kernel_initializer=cfg["kernel_initializer"],
                kernel_regularizer=cfg["kernel_regularizer"],
                use_bias=cfg["use_bias"],
                name=cfg["name"],
                kernel_quantizer="quantized_bits(8,0,1)",
                bias_quantizer="quantized_bits(8,0,1)"
            )

        elif isinstance(layer, Activation):
            return QActivation("quantized_relu(8,2)", name=cfg["name"])

        elif isinstance(layer, BatchNormalization):
            return QBatchNormalization(name=cfg["name"])

        else:
            return layer.__class__.from_config(cfg)

    print("\n--- Quantizing model using QKeras ---\n")
    quantized_model = clone_model(model, clone_function=convert_layer)
    quantized_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=3e-3),
        metrics=["accuracy"]
    )
    quantized_model.build(input_shape=model.input_shape)
    quantized_model.summary()
    return quantized_model

def prune_mlp_model(model, train_data, val_data, n_epochs=5):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=0,
            end_step=np.ceil(len(train_data) * n_epochs)
        )
    }

    pruned_model = prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    print("\n--- Training pruned MLP model ---\n")
    pruned_model.fit(train_data, validation_data=val_data,
                     epochs=n_epochs, callbacks=callbacks, verbose=2)

    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    print("\n--- Stripped final pruned MLP model ---\n")
    final_model.summary()

    return final_model
