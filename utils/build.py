from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model


    

def build_cnn(filter, neurons, input, n_classes=10):
    x = x_in = Input(input)
    for i, f in enumerate(filter):
        print(('Adding convolutional block {} with N={} filters').format(i, f))
        x = Conv2D(
            int(f),
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=False,
            name='conv_{}'.format(i),
        )(x)
        x = BatchNormalization(name='bn_conv_{}'.format(i))(x)
        x = Activation('relu', name='conv_act_%i' % i)(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(i))(x)
    x = Flatten()(x)

    for i, n in enumerate(neurons):
        print(('Adding dense block {} with N={} neurons').format(i, n))
        x = Dense(
            n, 
            kernel_initializer='lecun_uniform', 
            kernel_regularizer=l1(0.0001), 
            name='dense_%i' % i, 
            use_bias=False
        )(x)
        x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
        x = Activation('relu', name='dense_act_%i' % i)(x)
    x = Dense(int(n_classes), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='keras_baseline')
    model.summary()
    
    return model


from qkeras import QActivation, QDense, QConv2DBatchnorm
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

def build_q_cnn(filter, neurons, input, n_classes=10, quantizer_act='quantized_relu(6)', quantizer_dense="quantized_bits(6,0,alpha=1)", quantizer_conv="quantized_bits(6,0,alpha=1)"):
    x = x_in = Input(shape=input)

    # --- Convolutional blocks (q6.10 signed) ---
    for i, f in enumerate(filter):
        x = QConv2DBatchnorm(
            int(f),
            kernel_size=(3, 3),
            strides=(1, 1),
            
            kernel_quantizer=quantizer_conv,
            bias_quantizer=quantizer_conv,
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            use_bias=True,
            name=f'fused_convbn_{i}',
        )(x)
        # Quantized ReLU with same format
        x = QActivation(quantizer_act, name=f'conv_act_{i}')(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f'pool_{i}')(x)

    x = Flatten()(x)

    # --- Dense blocks (q6.10 signed) ---
    for i, n in enumerate(neurons):
        x = QDense(
            n,
            kernel_quantizer=quantizer_dense,
            bias_quantizer=None,                # keep BN after to absorb bias
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
            name=f'dense_{i}',
            use_bias=False,
        )(x)
        x = BatchNormalization(name=f'bn_dense_{i}')(x)
        x = QActivation(quantizer_act, name=f'dense_act_{i}')(x)

    # Τελικό layer (μπορεί να μείνει float για το softmax ή να γίνει QDense αν το θες 100% fixed-point)
    x = Dense(int(n_classes), name='output_dense')(x)
    x_out = Activation('softmax', name='output_softmax')(x)

    qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
    qmodel.summary()
    return qmodel


import os
import shutil
import subprocess
from pathlib import Path

def _run(cmd_list, cwd, log_prefix):
    """
    Run a command (list form), capture stdout/stderr into build_logs/,
    and raise with a concise error summary if it fails.
    """
    logs_dir = Path(cwd) / "build_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    out_f = logs_dir / f"{log_prefix}.stdout.txt"
    err_f = logs_dir / f"{log_prefix}.stderr.txt"

    print(f"\n[BUILD] Running: {' '.join(cmd_list)}\n        cwd={cwd}")
    with open(out_f, "w") as out, open(err_f, "w") as err:
        proc = subprocess.run(cmd_list, cwd=cwd, stdout=out, stderr=err, text=True)

    if proc.returncode != 0:
        # Prepare a short summary from the end of the logs
        def tail(p, n=80):
            try:
                txt = Path(p).read_text(errors="replace").splitlines()
                return "\n".join(txt[-n:])
            except Exception:
                return f"<could not read {p}>"

        out_tail = tail(out_f, 40)
        err_tail = tail(err_f, 80)

        # Pull out obvious error markers
        def grep_errors(s):
            lines = []
            for ln in s.splitlines():
                u = ln.upper()
                if ("ERROR" in u) or ("CRITICAL" in u) or ("FAILED" in u):
                    lines.append(ln)
            return "\n".join(lines[:30])  # cap
        highlights = grep_errors(out_tail + "\n" + err_tail)

        msg = (
            f"[BUILD] Command failed: {' '.join(cmd_list)}\n"
            f"[BUILD] Exit code: {proc.returncode}\n"
            f"[BUILD] Stdout tail ({out_f}):\n{out_tail}\n\n"
            f"[BUILD] Stderr tail ({err_f}):\n{err_tail}\n\n"
            f"[BUILD] Highlights:\n{highlights if highlights else '(no explicit ERROR lines captured)'}\n"
            f"[BUILD] Full logs saved in: {logs_dir}\n"
        )
        raise RuntimeError(msg)

def _which_or_none(name):
    p = shutil.which(name)
    return p if p else None

def _pick_hls_cmd():
    """
    Prefer vitis-run if available (Vitis 2024+), otherwise fall back to vitis_hls.
    Returns a (cmd_list, label) pair.
    """
    vr = _which_or_none("vitis-run")
    vh = _which_or_none("vitis_hls")
    if vr:
        # Vitis unified runner
        return (["vitis-run", "--mode", "hls", "--tcl", "build_prj.tcl"], "vitis-run")
    elif vh:
        # Classic HLS front-end
        return (["vitis_hls", "-f", "build_prj.tcl"], "vitis_hls")
    else:
        raise RuntimeError(
            "Neither 'vitis-run' nor 'vitis_hls' was found in PATH. "
            "Make sure your Vitis/Vitis HLS environment is sourced."
        )

def build_project(
    project_dir="Projects/Baseline",
    dataset="mnist",      # or "svhn"
    n_samples=100,
    seed=42,
    model_path="keras_model.keras",  # used by gen_tb.py
    ap_total_bits=16,
    ap_int_bits=6,
):
    """
    Build flow:
      1) Generate TB data (gen_tb.py)
      2) Run Vitis HLS (build_prj.tcl) via vitis-run or vitis_hls
      3) Generate golden predictions (golden_preds.py)

    Logs go to: <project_dir>/build_logs/
    """
    proj = Path(project_dir).resolve()

    # ---- preflight checks
    must_exist = [
        proj / "gen_tb.py",
        proj / "build_prj.tcl",
        proj / "golden_preds.py",
    ]
    missing = [str(p) for p in must_exist if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files:\n  - " + "\n  - ".join(missing))

    # 1) TB data
    tb_cmd = [
        "python", "gen_tb.py",
        "--dataset", dataset,
        "--n-samples", str(n_samples),
        "--seed", str(seed),
        "--ap-total-bits", str(ap_total_bits),
        "--ap-int-bits", str(ap_int_bits),
        "--model", model_path,
        "--save-labels",
    ]
    _run(tb_cmd, cwd=str(proj), log_prefix="01_gen_tb")

    # 2) HLS: pick the right frontend
    hls_cmd, hls_label = _pick_hls_cmd()
    _run(hls_cmd, cwd=str(proj), log_prefix=f"02_hls_{hls_label}")
    
    # 3) Golden preds
    _run(["python", "golden_preds.py"], cwd=str(proj), log_prefix="03_golden_preds")


    print("\n✅ Build completed successfully.")

# Example:
# build_project(project_dir="Projects/Baseline", dataset="svhn", n_samples=100)
