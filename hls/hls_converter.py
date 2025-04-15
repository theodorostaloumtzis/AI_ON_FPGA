# hls_converter.py
import os
import re
import pprint
import hls4ml
from hls4ml.report import read_vivado_report
from utils.plotting import print_dict
from utils.model_utils import save_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

def evaluate_model(model, test_data, do_bitstream=False, board_name="pynq-z2", part='xc7z020clg400-1', reuse=1.0, strat='Resource'):
    print("\n--- Evaluating/Saving Model ---\n")

    model_dir = "models"
    model_name = "keras_baseline"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    save_model(model, model_path)
    print(f"Model saved to {model_path}.h5")

    backend = "VivadoAccelerator" if do_bitstream else "Vitis"

    hls_config_aq = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_config_aq['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config_aq['Model']['Strategy'] = strat
    hls_config_aq['LayerName']['output_softmax'] = {'Strategy': 'Stable'}

    per_layer_reuse = reuse_percentage_to_factors(model, reuse)
    for layer, val in per_layer_reuse.items():
        hls_config_aq['LayerName'][layer] = {'ReuseFactor': val}

    print_dict(hls_config_aq)

    save_path = os.path.join("Projects", "AutoQKeras")
    os.makedirs(save_path, exist_ok=True)

    cfg_aq = hls4ml.converters.create_config(backend=backend)
    cfg_aq['IOType'] = 'io_stream'
    cfg_aq['HLSConfig'] = hls_config_aq
    cfg_aq['KerasModel'] = model
    cfg_aq['OutputDir'] = save_path
    cfg_aq['Board' if do_bitstream else 'XilinxPart'] = board_name if do_bitstream else part

    hls_model_aq = hls4ml.converters.keras_to_hls(cfg_aq)
    hls_model_aq.compile()
    print("hls4ml model compilation complete.")

    max_size, _ = calculate_max_hls_array_size(model)
    update_tcl_config(save_path, max(4096, max_size))
    return hls_model_aq, save_path

def reuse_percentage_to_factors(model, percent):
    from ..trainer.trainer import get_valid_reuse_factors_for_model
    valid_reuse_map = get_valid_reuse_factors_for_model(model)
    per_layer_reuse = {}
    for layer_name, reuse_list in valid_reuse_map.items():
        if reuse_list:
            index = int(percent * (len(reuse_list) - 1))
            per_layer_reuse[layer_name] = reuse_list[index]
    return per_layer_reuse

def update_tcl_config(project_dir: str, new_max_size: int, default_part: str = 'xc7z020clg400-1'):
    tcl_path = os.path.join(project_dir, 'project.tcl')
    if not os.path.exists(tcl_path):
        print(f"File not found: {tcl_path}")
        return

    with open(tcl_path, 'r') as file:
        lines = file.readlines()

    max_updated, part_updated = False, False
    for i, line in enumerate(lines):
        if re.match(r'^set\s+maximum_size\s+\d+', line):
            lines[i] = f"set maximum_size {new_max_size}\n"
            max_updated = True
        elif re.match(r'^set\s+part\s+\".*\"$', line):
            lines[i] = f'set part "{default_part}"\n'
            part_updated = True

    if not max_updated:
        lines.append(f"set maximum_size {new_max_size}\n")

    with open(tcl_path, 'w') as file:
        file.writelines(lines)

    print(f" maximum_size set to {new_max_size}")
    print(f" part set to '{default_part}'" if part_updated else f" part already matches '{default_part}'")

def calculate_max_hls_array_size(model: Model):
    max_size = 0
    layer_sizes = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            input_size = np.prod(layer.input_shape[1:])
            output_size = layer.units
            size = input_size * output_size
        elif isinstance(layer, tf.keras.layers.Conv2D):
            k = np.prod(layer.kernel_size)
            in_ch = int(layer.input_shape[-1])
            out_ch = int(layer.filters)
            size = k * in_ch * out_ch
        else:
            continue
        layer_sizes.append((layer.name, size))
        max_size = max(max_size, size)
    return max_size, layer_sizes

def finalize_hls_project(hls_model, project_dir, do_synth=False, do_report=False, do_bitstream=False):
    if do_synth:
        print("\n--- Running HLS Synthesis build (Vivado HLS) ---\n")
        hls_model.build(csim=False, synth=True, vsynth=True)
        if do_report:
            print("\n--- Reading Vivado Report (Synthesis) ---\n")
            pprint.pprint(read_vivado_report(project_dir))
    elif do_bitstream:
        print("\n--- Running HLS Build for Bitstream ---\n")
        hls_model.build(csim=False, export=True, bitfile=True)
        if do_report:
            print("\n--- Reading Vivado Report (Bitstream Build) ---\n")
            pprint.pprint(read_vivado_report(project_dir))
    else:
        print("\nNo synthesis or bitstream build requested. Done.")
