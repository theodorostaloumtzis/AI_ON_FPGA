from pathlib import Path
from tensorflow.keras import callbacks, optimizers, losses, metrics
from ai_on_fpga.utils import log

def fit(model, ds_train, ds_val, cfg: dict, out_dir: Path):
    model.compile(
        optimizer = optimizers.Adam(cfg.get("lr", 1e-3)),
        loss      = losses.SparseCategoricalCrossentropy(),
        metrics   = [metrics.SparseCategoricalAccuracy()]
    )

    ckpt = callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best.h5"),
        save_best_only=True,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
    )

    model.fit(
        ds_train,
        validation_data = ds_val,
        epochs          = cfg.get("epochs", 3),
        callbacks       = [ckpt],
    )
    log.info("Training complete – best model saved to %s", out_dir / "best.h5")
    return model
