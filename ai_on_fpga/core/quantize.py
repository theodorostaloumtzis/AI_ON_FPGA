"""
Pruning + QKeras post-training quantisation.

Reads its behaviour from cfg['quantize']:

quantize:
  enabled:        true
  prune:
    initial:      0.0
    final:        0.50
    begin_epoch:  2
    end_epoch:    10
  qkeras:
    bits:         6          # quantized_bits(bits, 0)
    alpha:        1
"""

from pathlib import Path
from typing import Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import (
    PolynomialDecay,
    prune_low_magnitude,
    strip_pruning,
)
from qkeras import QDense, QActivation
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras import layers, models, regularizers

from ai_on_fpga.utils import log


def _apply_pruning(model: tf.keras.Model, cfg: dict, steps_per_epoch: int):
    """Wrap Dense/Conv2D layers with pruning."""
    begin = steps_per_epoch * cfg["begin_epoch"]
    end = steps_per_epoch * cfg["end_epoch"]
    schedule = PolynomialDecay(
        initial_sparsity=cfg["initial"],
        final_sparsity=cfg["final"],
        begin_step=begin,
        end_step=end,
        frequency=steps_per_epoch,
    )

    def _prune_fn(layer):
        if isinstance(layer, (layers.Conv2D, layers.Dense)) and layer.name != "output":
            return prune_low_magnitude(layer, pruning_schedule=schedule)
        return layer

    return tf.keras.models.clone_model(model, clone_function=_prune_fn)


def _build_qkeras_mlp(input_shape, n_classes, dense_sizes, bits, alpha):
    """Manual QKeras MLP like the one in build.py."""
    x_in = layers.Input(shape=input_shape, name="input")
    x = x_in
    for i, n in enumerate(dense_sizes):
        x = QDense(
            n,
            kernel_quantizer=f"quantized_bits({bits}, 0, alpha={alpha})",
            bias_quantizer=f"quantized_bits({bits}, 0, alpha={alpha})",
            kernel_initializer="lecun_uniform",
            kernel_regularizer=regularizers.l1(1e-4),
            use_bias=True,
            name=f"qdense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = QActivation(f"quantized_relu({bits})", name=f"qact_{i}")(x)
    x = QDense(n_classes, name="output_dense")(x)
    out = layers.Activation("softmax", name="output")(x)
    return models.Model(x_in, out, name="qkeras_mlp")


def apply(
    model: tf.keras.Model,
    cfg: dict,
    out_dir: Path,
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Returns (baseline_model, quantised_model)

    * baseline_model  – pruned + stripped
    * quantised_model – optional QKeras model (may be None)
    """
    if not cfg.get("enabled", False):
        log.info("Quantisation disabled – returning original model")
        return model, None

    # 1. PRUNING ------------------------------------------------------------
    log.info("⏳  Pruning enabled – cloning model with pruning wrappers")
    steps = tf.data.experimental.cardinality(ds_train).numpy()
    pruned = _apply_pruning(model, cfg["prune"], steps)

    pruned.compile(
        optimizer=tf.keras.optimizers.Adam(3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    pruned.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.get("epochs", 30),
        callbacks=callbacks,
        verbose=1,
    )
    pruned = strip_pruning(pruned)
    pruned_path = out_dir / "pruned_model.h5"
    pruned.save(pruned_path)
    log.info("✅  Pruned model saved → %s", pruned_path)

    # 2. OPTIONAL QKERAS -----------------------------------------------------
    q_cfg = cfg.get("qkeras", {})
    if not q_cfg:
        return pruned, None

    log.info("⏳  Building %d-bit QKeras model", q_cfg["bits"])
    qmodel = _build_qkeras_mlp(
        input_shape=(784,),
        n_classes=10,
        dense_sizes=[64, 32],
        bits=q_cfg["bits"],
        alpha=q_cfg.get("alpha", 1),
    )
    qmodel.compile(
        optimizer=tf.keras.optimizers.Adam(3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    qmodel.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.get("epochs", 30),
        callbacks=callbacks,
        verbose=1,
    )
    qmodel = strip_pruning(qmodel)
    q_path = out_dir / "quantized_model.h5"
    qmodel.save(q_path)
    log.info("✅  Quantised model saved → %s", q_path)

    return pruned, qmodel
