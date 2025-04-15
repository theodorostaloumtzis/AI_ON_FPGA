# main.py
from datetime import datetime
import sys
import os

def init_run_log():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = open(log_path, 'w')
    sys.stderr = sys.stdout
    print(f"[INFO] Logging this run to {log_path}")

import argparse
from config.config import setup_environment
from data.data_pipeline import prepare_data
from trainer.trainer import train_model, prune_mlp_model, quantize_model
from quant.autoqkeras_utils import run_autoqkeras_tuning, process_best_autoqkeras_model
from hls.hls_converter import evaluate_model, finalize_hls_project
from utils.model_manager import ModelManager

def main():
    init_run_log()

    strategy_full_help = (
        "Strategy determines the hardware optimization approach used during High-Level Synthesis (HLS). "
        "It controls how the neural network is translated into digital logic, specifically balancing between latency, "
        "resource usage, and performance.\n\n"
        "Available Strategy options:\n"
        "  - Latency:   Minimize inference time by increasing parallelism and pipelining.\n"
        "  - Resource(Default):  Minimize FPGA resource usage by reusing computation units.\n"
        "  - Balanced:  Strike a balance between resource usage and latency.\n"
        "  - None:      Use the default strategy as determined by the synthesis tool."
    )

    parser = argparse.ArgumentParser(description="Train, evaluate, optionally run AutoQKeras, and optionally do HLS synthesis or bitstream generation.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--bitstream", action="store_true")
    parser.add_argument("--board", type=str, default="ZCU104")
    parser.add_argument("--autoqk", action="store_true")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--reuse", type=float, default=1.0)
    parser.add_argument("--strat", type=str, default="Resource", help=strategy_full_help)
    parser.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "mlp"])

    args = parser.parse_args()
    if args.synth and args.bitstream:
        print("\nERROR: --synth and --bitstream cannot both be used in the same run.")
        return

    setup_environment()
    train_data, val_data, test_data = prepare_data()
    model = ModelManager(model_type=args.model_type).build_model()
    model = train_model(model, train_data, val_data, test_data, n_epochs=args.epochs)

    if not args.autoqk:
        if args.model_type == "mlp":
            model = prune_mlp_model(model, train_data, val_data, n_epochs=5)
        model = quantize_model(model)
    else:
        autoqk = run_autoqkeras_tuning(model, train_data, val_data, n_epochs=args.epochs, max_trials=args.max_trials, model_type=args.model_type)
        model = process_best_autoqkeras_model(autoqk.get_best_model(), train_data, val_data, test_data, args.epochs, args.model_type)

    hls_model, hls_path = evaluate_model(model, test_data, do_bitstream=args.bitstream, board_name=args.board, reuse=args.reuse, strat=args.strat)
    finalize_hls_project(hls_model, hls_path, do_synth=args.synth, do_report=args.report, do_bitstream=args.bitstream)


if __name__ == "__main__":
    main()