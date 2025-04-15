# autoqkeras_utils.py
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from qkeras.autoqkeras import AutoQKeras

def run_autoqkeras_tuning(model, train_data, val_data, n_epochs=10, max_trials=5, model_type="cnn"):
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

    limit = get_autoqkeras_limits(model)

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
        "mode": "bayesian",
        "seed": 42,
        "limit": limit,
        "tune_filters": "layer",
        "tune_filters_exceptions": "^output",
        "distribution_strategy": None,
        "max_trials": max_trials,
    }

    autoqk = AutoQKeras(model=model, output_dir="autoqk_results", **run_config)

    print("\n--- Running AutoQKeras search ---\n")
    autoqk.fit(x=train_data, validation_data=val_data, epochs=n_epochs)
    return autoqk

def get_autoqkeras_limits(model):
    limit = {"conv": [], "dense": [], "act": []}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            limit["conv"].append(layer.filters)
        elif isinstance(layer, tf.keras.layers.Dense):
            limit["dense"].append(layer.units)
        elif isinstance(layer, tf.keras.layers.Activation):
            limit["act"].append(16)  # default max act width
    return limit

def process_best_autoqkeras_model(best_model, train_data, val_data, test_data, n_epochs, model_type="cnn"):
    print("\n--- Processing Best AutoQKeras Model ---\n")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]

    start = time.time()
    best_model.fit(train_data, epochs=n_epochs, validation_data=val_data, callbacks=callbacks, verbose=1)
    end = time.time()
    print(f'\nTraining completed in {(end - start) / 60.0:.2f} minutes\n')

    weight_file = f"autoqkeras_{model_type}_weights.h5"
    best_model.save_weights(weight_file)

    layers = [l for l in best_model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        x = layers[i](x)

    new_model = Model(inputs=[layers[0].input], outputs=[x])
    new_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        metrics=["accuracy"]
    )
    new_model.load_weights(weight_file)

    results = new_model.evaluate(test_data, verbose=2)
    metrics = dict(zip(new_model.metrics_names, results))
    print("\nAutoQKeras best model test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return new_model