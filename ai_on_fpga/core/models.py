"""
Two toy architectures: small CNN (default) or a single-hidden-layer MLP.
"""
from tensorflow.keras import layers, models

def build(cfg: dict):
    net_type = cfg.get("type", "cnn")
    hidden   = cfg.get("hidden", 128)

    if net_type == "cnn":
        inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(16, 3, activation="relu")(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(hidden, activation="relu")(x)
    else:  # "mlp"
        inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Flatten()(inputs)
        x = layers.Dense(hidden, activation="relu")(x)

    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs, outputs)
