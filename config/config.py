# config.py
import os

def setup_environment():
    """
    Sets up environment variables needed for Vitis/Vivado toolchains.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2024.2:/tools/Xilinx/Vitis/2020.1/'
    os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
    os.environ['PATH'] = '/tools/Xilinx/Vitis_HLS/2024.2/bin:' + os.environ['PATH']
    os.environ['PATH'] = '/tools/Xilinx/Vitis/2020.1/bin:' + os.environ['PATH']


import yaml
import os

def load_model_config(path="model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)