# main.py
import argparse
import tensorflow as tf
from config.config import setup_environment, load_model_config
from data.data_pipeline import prepare_data
from trainer.trainer import train_model
from quant.autoqkeras_utils import run_autoqkeras_tuning, process_best_autoqkeras_model
from hls.hls_converter import evaluate_model, finalize_hls_project
from utils.model_manager import ModelManager

def main():
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
    parser.add_argument("--board", type=str, default="pynq-z2")
    parser.add_argument("--autoqk", action="store_true")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--reuse", type=float, default=1.0)
    parser.add_argument("--strat", type=str, default="Resource", help=strategy_full_help)
    parser.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--model-path", type=str, default=None, help="Path to a pre-trained Keras model to load.")

    args = parser.parse_args()

    if args.synth and args.bitstream:
        print("\nERROR: --synth and --bitstream cannot both be used in the same run.")
        return

    setup_environment()
    train_data, val_data, test_data = prepare_data()

    loaded_model = False

    if args.model_path:
        print(f"\n--- Loading model from {args.model_path} ---\n")
        model = tf.keras.models.load_model(args.model_path, compile=False)
        loaded_model = True
    else:
        cfg = load_model_config(path="model_conf.yaml")
        model_manager = ModelManager(cfg)
        model = model_manager.build_model()

    model = train_model(model, train_data, val_data, test_data, n_epochs=args.epochs)

    if not loaded_model:
        if not args.autoqk:
            from trainer.trainer import prune_mlp_model, quantize_model
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