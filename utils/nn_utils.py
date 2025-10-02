import json
import os
import pickle as pkl
import random
from io import BytesIO
from pathlib import Path
from typing import Callable

import h5py as h5
import numpy as np
import tensorflow as tf
import zstd
from HGQ.bops import trace_minmax
from keras.layers import Dense
from keras.src.layers.convolutional.base_conv import Conv
from keras.src.saving.legacy import hdf5_format
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm.auto import tqdm


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):  # type: ignore
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class SaveTopN(keras.callbacks.Callback):
    def __init__(
        self,
        metric_fn: Callable[[dict], float],
        n: int,
        path: str | Path,
        side: str = 'max',
        fname_format='epoch={epoch}-metric={metric:.4e}.h5',
        cond_fn: Callable[[dict], bool] = lambda x: True,
    ):
        self.n = n
        self.metric_fn = metric_fn
        self.path = Path(path)
        self.fname_format = fname_format
        os.makedirs(path, exist_ok=True)
        self.weight_paths = np.full(n, '/dev/null', dtype=object)
        if side == 'max':
            self.best = np.full(n, -np.inf)
            self.side = np.greater
        elif side == 'min':
            self.best = np.full(n, np.inf)
            self.side = np.less
        self.cond = cond_fn

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        assert isinstance(self.model, keras.models.Model)
        logs = logs.copy()
        logs['epoch'] = epoch
        if not self.cond(logs):
            return
        metric = self.metric_fn(logs)

        if self.side(metric, self.best[-1]):
            try:
                os.remove(self.weight_paths[-1])
            except OSError:
                pass
            logs['metric'] = metric
            fname = self.path / self.fname_format.format(**logs)
            self.best[-1] = metric
            self.weight_paths[-1] = fname
            self.model.save_weights(fname)
            with h5.File(fname, 'r+') as f:
                log_str = json.dumps(logs, cls=NumpyFloatValuesEncoder)
                f.attrs['train_log'] = log_str
            idx = np.argsort(self.best)
            if self.side == np.greater:
                idx = idx[::-1]
            self.best = self.best[idx]
            self.weight_paths = self.weight_paths[idx]

    def rename_ckpts(self, dataset, bsz=65536):
        assert self.weight_paths[0] != '/dev/null', 'No checkpoints to rename'
        assert isinstance(self.model, keras.models.Model)

        weight_buf = BytesIO()
        with h5.File(weight_buf, 'w') as f:
            hdf5_format.save_weights_to_hdf5_group(f, self.model)
        weight_buf.seek(0)

        for i, path in enumerate(tqdm(self.weight_paths, desc='Renaming checkpoints')):
            if path == '/dev/null':
                continue
            self.model.load_weights(path)
            bops = trace_minmax(self.model, dataset, bsz=bsz, verbose=False)
            with h5.File(path, 'r+') as f:
                logs = json.loads(f.attrs['train_log'])  # type: ignore
                logs['bops'] = bops
                metric = self.metric_fn(logs)
                logs['metric'] = metric
                f.attrs['train_log'] = json.dumps(logs, cls=NumpyFloatValuesEncoder)
            self.best[i] = metric
            new_fname = self.path / self.fname_format.format(**logs)
            os.rename(path, new_fname)
            self.weight_paths[i] = new_fname

        idx = np.argsort(self.best)
        self.best = self.best[idx]
        self.weight_paths = self.weight_paths[idx]
        with h5.File(weight_buf, 'r') as f:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, self.model)


class PBarCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric='loss: {loss:.2f}/{val_loss:.2f}'):
        self.pbar = None
        self.template = metric

    def on_epoch_begin(self, epoch, logs=None):
        if self.pbar is None:
            self.pbar = tqdm(total=self.params['epochs'], unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.pbar, tqdm)
        assert isinstance(logs, dict)
        self.pbar.update(1)
        string = self.template.format(**logs)
        if 'bops' in logs:
            string += f' - BOPs: {logs["bops"]:,.0f}'
        self.pbar.set_description(string)

    def on_train_end(self, logs=None):
        if self.pbar is not None:
            self.pbar.close()


def plot_history(histry: dict, metrics=('loss', 'val_loss'), ylabel='Loss', logy=False):
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.plot(histry[metric], label=metric)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()
    return fig, ax


def save_model(model: keras.models.Model, path: str):
    _path = Path(path)
    model.save(path)
    if model.history is not None:
        history = model.history.history
    else:
        history = {}
    with open(_path.with_suffix('.history'), 'wb') as f:
        f.write(zstd.compress(pkl.dumps(history)))


def load_model(path: str, co=None):
    _path = Path(path)
    model: keras.Model = keras.models.load_model(path, custom_objects=co)  # type: ignore
    with open(_path.with_suffix('.history'), 'rb') as f:
        history: dict[str, list] = pkl.loads(zstd.decompress(f.read()))
    return model, history


def save_history(history, path):
    with open(path, 'wb') as f:
        f.write(zstd.compress(pkl.dumps(history)))


def load_history(path):
    with open(path, 'rb') as f:
        history = pkl.loads(zstd.decompress(f.read()))
    return history


def absorb_batchNorm(model_target, model_original):
    for layer in model_target.layers:
        if layer.__class__.__name__ == 'Functional':
            absorb_batchNorm(layer, model_original.get_layer(layer.name))
            continue
        if (
            (isinstance(layer, Dense) or isinstance(layer, Conv))
            and len(nodes := model_original.get_layer(layer.name)._outbound_nodes) > 0
            and isinstance(nodes[0].outbound_layer, keras.layers.BatchNormalization)
        ):
            _gamma, _beta, _mu, _var = model_original.get_layer(layer.name)._outbound_nodes[0].outbound_layer.get_weights()
            _ratio = _gamma / np.sqrt(0.001 + _var)
            _bias = -_gamma * _mu / np.sqrt(0.001 + _var) + _beta

            k, *_b = model_original.get_layer(layer.name).get_weights()
            if _b:
                b = _b[0]
            else:
                b = np.zeros(layer.output_shape[-1])
            nk = np.einsum('...c, c-> ...c', k, _ratio, optimize=True)
            nb = np.einsum('...c, c-> ...c', b, _ratio, optimize=True) + _bias
            extras = layer.get_weights()[2:]
            layer.set_weights([nk, nb, *extras])
        elif hasattr(layer, 'kernel'):
            for w in layer.weights:
                if '_bw' not in w.name:
                    break
            else:
                continue
            weights = layer.get_weights()
            new_weights = model_original.get_layer(layer.name).get_weights()
            l = len(new_weights)  # noqa: E741 # If l looks like 1 by any chance, change your font.
            layer.set_weights([*new_weights, *weights[l:]][: len(weights)])


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    tf.config.experimental.enable_op_determinism()


def get_best_ckpt(save_path: Path, take_min=False):
    ckpts = list(save_path.glob('*.h5'))

    def rank(ckpt: Path):
        with h5.File(ckpt, 'r') as f:
            log: dict = f.attrs['train_log']  # type: ignore
        log = json.loads(log)  # type: ignore
        metric = log['metric']  # type: ignore
        return metric

    ckpts = sorted(ckpts, key=rank, reverse=not take_min)
    ckpt = ckpts[0]
    return ckpt


class PeratoFront(keras.callbacks.Callback):
    def __init__(
        self,
        path: str | Path,
        fname_format: str,
        metrics_names: list[str],
        sides: list[int],
        cond_fn: Callable[[dict], bool] = lambda x: True,
    ):
        self.path = Path(path)
        self.fname_format = fname_format
        os.makedirs(path, exist_ok=True)
        self.paths = []
        self.metrics = []
        self.metric_names = metrics_names
        self.sides = np.array(sides)
        self.cond_fn = cond_fn

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(self.model, keras.models.Model)
        assert isinstance(logs, dict)

        logs = logs.copy()
        logs['epoch'] = epoch

        if not self.cond_fn(logs):
            return
        new_metrics = np.array([logs[metric_name] for metric_name in self.metric_names])
        _rm_idx = []
        for i, old_metrics in enumerate(self.metrics):
            _old_metrics = self.sides * old_metrics
            _new_metrics = self.sides * new_metrics
            if np.all(_new_metrics <= _old_metrics):
                return
            if np.all(_new_metrics >= _old_metrics):
                _rm_idx.append(i)
        for i in _rm_idx[::-1]:
            self.metrics.pop(i)
            p = self.paths.pop(i)
            os.remove(p)

        path = self.path / self.fname_format.format(**logs)
        self.metrics.append(new_metrics)
        self.paths.append(path)
        self.model.save_weights(self.paths[-1])

        with h5.File(path, 'r+') as f:
            log_str = json.dumps(logs, cls=NumpyFloatValuesEncoder)
            f.attrs['train_log'] = log_str

    def rename_ckpts(self, dataset, bsz=65536):
        assert isinstance(self.model, keras.models.Model)

        weight_buf = BytesIO()
        with h5.File(weight_buf, 'w') as f:
            hdf5_format.save_weights_to_hdf5_group(f, self.model)
        weight_buf.seek(0)

        for i, path in enumerate(tqdm(self.paths, desc='Renaming checkpoints')):
            self.model.load_weights(path)
            bops = trace_minmax(self.model, dataset, bsz=bsz, verbose=False)
            with h5.File(path, 'r+') as f:
                logs = json.loads(f.attrs['train_log'])  # type: ignore
                logs['bops'] = bops
                f.attrs['train_log'] = json.dumps(logs, cls=NumpyFloatValuesEncoder)
                metrics = np.array([logs[metric_name] for metric_name in self.metric_names])
            self.metrics[i] = metrics
            new_fname = self.path / self.fname_format.format(**logs)
            os.rename(path, new_fname)
            self.paths[i] = new_fname

        with h5.File(weight_buf, 'r') as f:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, self.model)


class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_fn: Callable[[int], float]):
        self.beta_fn = beta_fn

    def on_epoch_begin(self, epoch, logs=None):
        assert isinstance(self.model, keras.models.Model)

        beta = self.beta_fn(epoch)
        for layer in self.model.layers:
            if hasattr(layer, 'beta'):
                layer.beta.assign(keras.backend.constant(beta, dtype=keras.backend.floatx()))

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        logs['beta'] = self.beta_fn(epoch)

    @classmethod
    def from_config(cls, config):
        return cls(get_schedule(config.beta, config.train.epochs))


def get_schedule(beta_conf, total_epochs):
    epochs = []
    betas = []
    interpolations = []
    for block in beta_conf.intervals:
        epochs.append(block.epochs)
        betas.append(block.betas)
        interpolation = block.interpolation
        assert interpolation in ['linear', 'log']
        interpolations.append(interpolation == 'log')
    epochs = np.array(epochs + [total_epochs])
    assert np.all(np.diff(epochs) >= 0)
    betas = np.array(betas)
    interpolations = np.array(interpolations)

    def schedule(epoch):
        if epoch >= total_epochs:
            return betas[-1, -1]
        idx = np.searchsorted(epochs, epoch, side='right') - 1
        beta0, beta1 = betas[idx]
        epoch0, epoch1 = epochs[idx], epochs[idx + 1]
        if interpolations[idx]:
            beta = beta0 * (beta1 / beta0) ** ((epoch - epoch0) / (epoch1 - epoch0))
        else:
            beta = beta0 + (beta1 - beta0) * (epoch - epoch0) / (epoch1 - epoch0)
        return float(beta)

    return schedule


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def doWeights(model):
    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch") != -1 or len(layer.get_weights()) < 1:
            continue
        weights = layer.weights[0].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name, np.sum(weights == 0) / np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10, 10))
    bins = np.linspace(-1.5, 1.5, 50)
    plt.hist(histosW, bins, histtype='stepfilled', stacked=True, label=labelsW, edgecolor='black')
    plt.legend(frameon=False, loc='upper left')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    plt.figtext(0.2, 0.38, model._name, wrap=True, horizontalalignment='left', verticalalignment='center')


import math
from collections import defaultdict

def reuse_percentage_to_factors(model, serial_pct: float = 1.0):
    """Convert a serial percentage to reuse factors for each layer in a model."""
    from tensorflow.keras.layers import Dense, Conv2D, DepthwiseConv2D
    from qkeras import QDense, QConv2D, QDepthwiseConv2D

    if not 0.0 <= serial_pct <= 1.0:
        raise ValueError("serial_pct must be in [0, 1]")

    reuse = {}

    
    def legal_divisor(n, candidate):
        """Decrease candidate until it cleanly divides n."""
        while candidate > 1 and n % candidate:
            candidate -= 1
        return max(1, candidate)
    

    for layer in model.layers:
        if isinstance(layer, (Dense, QDense)):
            total = layer.input_shape[-1] * layer.units          # Nin × Nout

        elif isinstance(layer, (Conv2D, QConv2D)):
            kh, kw   = layer.kernel_size
            cin      = layer.input_shape[-1]
            cout     = layer.filters
            total    = kh * kw * cin * cout                      # per-pixel MACs

        elif isinstance(layer, (DepthwiseConv2D, QDepthwiseConv2D)):
            kh, kw   = layer.kernel_size
            cin      = layer.input_shape[-1]
            total    = kh * kw * cin                             # per-pixel MACs

        else:
            continue  # no MACs → nothing to tune

        #   serial_pct = 1.0  → target_rf = total  (most serial)
        #   serial_pct = 0.0  → target_rf = 1      (fully parallel)
        target_rf = int(round(total * serial_pct))
        target_rf = min(max(1, target_rf), total)

        # Snap downward to the nearest divisor so that total % rf == 0
        rf = legal_divisor(total, target_rf)

        reuse[layer.name] = rf
        print(f"{layer.name:20s}  MACs={total:6d}  ReuseFactor={rf}")

    return reuse

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer, steps=0):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.50, begin_step=steps * 2, end_step=steps * 10, frequency=steps
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer


import os, time, math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow_model_optimization.sparsity import keras as pruning

# ------------------------------------------------------------
# REQUIRED USER HOOKS
# ------------------------------------------------------------
# 1) Provide a fresh model each fold:
#    def build_model():
#        return <your pruned/quantized model instance>
#
# 2) Provide data as X, y (NumPy arrays or TF tensors)
#    shapes: X -> (N, H, W, C) or (N, features...), y -> (N,) or one-hot (N, num_classes)
#    If y is one-hot, it will be argmaxed for stratification/class_weight.

def run_kfold_training(
    build_model,
    X, y,
    n_splits=5,
    epochs=50,
    batch_size=128,
    save_dir="models",
    base_name="baseline_svhn",
    learning_rate=3e-3,
    patience_es=10,
    patience_rlrop=3,
    class_weight_external=None,
    shuffle_each_fold=True,
    seed=42,
    prefetch=True,
    train=True,
    save_all_folds=False,   # <-- NEW
):
    import os, time, numpy as np, tensorflow as tf
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight

    os.makedirs(save_dir, exist_ok=True)
    save_path_prunned = os.path.join(save_dir, f"{base_name}.h5")

    y_strat = np.argmax(y, axis=-1) if (y.ndim > 1 and y.shape[-1] > 1) else y
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_histories, fold_val_acc = [], []
    best_val_acc = -np.inf
    best_fold = None

    if train:
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y_strat), start=1):
            print(f"\n====== Fold {fold_idx}/{n_splits} ======")

            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            if shuffle_each_fold:
                train_ds = train_ds.shuffle(min(len(X_tr), 10_000), seed=seed, reshuffle_each_iteration=True)
            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            # class weights
            if class_weight_external is not None:
                class_weight_fold = class_weight_external
            else:
                classes = np.unique(y_strat)
                cw = compute_class_weight("balanced", classes=classes, y=y_strat[tr_idx])
                class_weight_fold = {int(c): float(w) for c, w in zip(classes, cw)}

            model = build_model()
            LOSS = tf.keras.losses.CategoricalCrossentropy()
            OPT  = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
            model.compile(loss=LOSS, optimizer=OPT, metrics=["accuracy"])

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if model.metrics_names and "val_loss" in model.metrics_names else "loss",
                    patience=patience_es, verbose=1, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=patience_rlrop, verbose=1),
                pruning_callbacks.UpdatePruningStep(),
            ]

            start = time.time()
            # if you removed validation from fit, use: validation_data=None and callbacks monitor "loss"
            history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,        # remove if you truly have no val during fit
                callbacks=callbacks,
                class_weight=class_weight_fold,
                verbose=1
            )
            print(f"[Fold {fold_idx}] {(time.time()-start)/60.0:.2f} min")

            # evaluate on fold holdout
            val_loss, val_acc = model.evaluate(val_ds, verbose=0)
            fold_val_acc.append(float(val_acc))
            fold_histories.append(history.history)
            print(f"[Fold {fold_idx}] val_acc={val_acc:.4f}")

            # ---- SAVE LOGIC ----
            if save_all_folds:
                fold_path = os.path.join(save_dir, f"{base_name}_fold{fold_idx}.h5")
                model.save(fold_path)
                print(f"[Fold {fold_idx}] saved {fold_path}")

            # keep only best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold = fold_idx
                model.save(save_path_prunned)   # overwrite best
                print(f"[BEST] fold {best_fold} now best (val_acc={best_val_acc:.4f}); "
                      f"saved -> {save_path_prunned}")

        print(f"\n[KFold] Best fold={best_fold} val_acc={best_val_acc:.4f}")
        return {
            "fold_val_acc": fold_val_acc,
            "histories": fold_histories,
            "best_fold": best_fold,
            "canonical_model_path": save_path_prunned
        }

    else:
        from qkeras.utils import _add_supported_quantized_objects
        from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
        co = {}; _add_supported_quantized_objects(co); co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
        print(f"[LOAD] {save_path_prunned}")
        model = tf.keras.models.load_model(save_path_prunned, custom_objects=co)
        return {"model": model, "loaded_from": save_path_prunned}



# ---------------------------------------------------------------------
# OPTIONAL: if your data pipeline is already tf.data.Dataset-based and
# you *don’t* have X/y arrays handy, here’s a helper to extract arrays
# once (only suitable when the dataset fits in memory):
# ---------------------------------------------------------------------
def dataset_to_arrays(dataset, limit=None):
    xs, ys = [], []
    for i, (xb, yb) in enumerate(dataset):
        xs.append(xb.numpy())
        ys.append(yb.numpy())
        if limit is not None and (i+1) >= limit:
            break
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


# ---------------------------
# EXAMPLE USAGE (sketch):
# ---------------------------
# def build_model():
#     # create your pruned/quantized model here (fresh instance per fold)
#     return model_pruned_instance
#
# # If you currently have train_data/val_data as tf.data pipelines,
# # first materialize a full X, y once (e.g., from your combined/whole dataset):
# # X, y = dataset_to_arrays(full_dataset)  # full_dataset should cover all samples
#
# results = run_kfold_training(
#     build_model=build_model,
#     X=X,
#     y=y,
#     n_splits=5,
#     epochs=n_epochs,
#     batch_size=128,
#     save_dir=models_path,
#     base_name="baseline_svhn",
#     learning_rate=3e-3,
#     patience_es=10,
#     patience_rlrop=3,
#     class_weight_external=None,   # or pass your precomputed dict
#     train=train
# )
#
# # When train=False (later):
# # loaded = run_kfold_training(build_model, X, y, train=False)
# # model_pruned = loaded["model"]


import json
import os
import numpy as np
import tensorflow as tf

def _to_jsonable(obj):
    """Recursively convert NumPy/TF objects into JSON-serializable Python types."""
    # NumPy scalars
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # TF tensors
    if isinstance(obj, tf.Tensor):
        return _to_jsonable(obj.numpy())
    # Containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # Bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    # Already JSON-friendly (int, float, bool, str, None)
    return obj

def save_kfold_results(results, path):
    """
    results: dict with keys like 'histories' (list of History.history dicts),
             'fold_val_acc' (list of floats), etc.
    path: output file (e.g., ".../baseline_svhn_kfold.history" or ".json")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "histories": results.get("histories", []),
        "fold_val_acc": results.get("fold_val_acc", []),
    }
    payload = _to_jsonable(payload)  # <-- sanitize everything
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] wrote: {path}")

def load_kfold_results(path):
    with open(path, "r") as f:
        return json.load(f)

def available_metrics(histories):
    """Return the intersection of metric keys across folds (non-empty)."""
    if not histories:
        return []
    common = set(histories[0].keys())
    for h in histories[1:]:
        common &= set(h.keys())
    return sorted(common)

def plot_histories_per_fold(histories, metrics=("loss",), title_prefix="Fold", alpha=0.85):
    """
    Plot each fold’s curve for the requested metrics, skipping any metric
    that’s not present in a history dict.
    """
    for metric in metrics:
        plt.figure()
        plotted = False
        for i, h in enumerate(histories, start=1):
            if metric in h:
                y = np.array(h[metric], dtype=float)
                x = np.arange(1, len(y) + 1)
                plt.plot(x, y, label=f"{title_prefix} {i}", alpha=alpha)
                plotted = True
        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} per fold")
            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.show()
        else:
            print(f"[plot] skipped '{metric}' (not found in any history)")

def plot_histories_mean_std(histories, metrics=("loss",), label="mean±std", color=None):
    """
    Plot mean ± std across folds for each requested metric.
    Only folds that contain the metric are used.
    Handles different epoch lengths by aligning to the shortest length.
    """
    for metric in metrics:
        # collect series with this metric
        series = [np.array(h[metric], dtype=float) for h in histories if metric in h]
        if not series:
            print(f"[plot] skipped '{metric}' (not found)")
            continue
        # align to shortest length
        min_len = min(map(len, series))
        if min_len == 0:
            print(f"[plot] skipped '{metric}' (empty series)")
            continue
        arr = np.stack([s[:min_len] for s in series], axis=0)  # [folds, epochs]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        x = np.arange(1, min_len + 1)

        plt.figure()
        line, = plt.plot(x, mean)
        c = color if color is not None else line.get_color()
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label=label, color=c)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} (mean ± std across folds)")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


import math
from collections import defaultdict

def reuse_factors_with_serial_pct(model):
    from tensorflow.keras.layers import Dense, Conv2D, DepthwiseConv2D
    from qkeras import QDense, QConv2D, QDepthwiseConv2D

    def get_total_macs(layer):
        if isinstance(layer, (Dense, QDense)):
            return layer.input_shape[-1] * layer.units
        elif isinstance(layer, (Conv2D, QConv2D)):
            kh, kw = layer.kernel_size
            cin = layer.input_shape[-1]
            cout = layer.filters
            return kh * kw * cin * cout
        elif isinstance(layer, (DepthwiseConv2D, QDepthwiseConv2D)):
            kh, kw = layer.kernel_size
            cin = layer.input_shape[-1]
            return kh * kw * cin
        else:
            return None  # Unsupported layer

    def get_divisors(n):
        """Return all positive integers that divide n."""
        return sorted({i for i in range(1, n + 1) if n % i == 0})

    for layer in model.layers:
        total_macs = get_total_macs(layer)
        if total_macs is None:
            continue

        print(f"\nLayer: {layer.name} ({layer.__class__.__name__}) - Total MACs: {total_macs}")
        print(f"{'ReuseFactor':>12} | {'SerialPct':>10}")
        print("-" * 27)

        for rf in get_divisors(total_macs):
            serial_pct = rf / total_macs
            print(f"{rf:12d} | {serial_pct:10.4f}")
