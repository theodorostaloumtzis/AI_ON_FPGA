#!/usr/bin/env python3
# accuracy.py
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Input files – change here if you keep them elsewhere
# ------------------------------------------------------------------
PRED_FILE = Path("tb_data/rtl_cosim_results.log")          # DUT output
GOLD_FILE = Path("tb_data/tb_output_predictions.dat") # golden model
# ------------------------------------------------------------------

def load_matrix(path: Path) -> np.ndarray:
    """Load whitespace-separated floats into a 2-D numpy array."""
    with path.open() as f:
        data = [list(map(float, line.split())) for line in f if line.strip()]
    return np.asarray(data, dtype=np.float32)

def main():
    preds = load_matrix(PRED_FILE)
    gold  = load_matrix(GOLD_FILE)

    if preds.shape != gold.shape:
        raise ValueError(f"Shape mismatch: {preds.shape} vs {gold.shape}")

    pred_labels = preds.argmax(axis=1)
    gold_labels = gold.argmax(axis=1)

    correct = (pred_labels == gold_labels).sum()
    accuracy = correct / len(pred_labels)

    # build confusion matrix (rows = true label, cols = predicted)
    cm = np.zeros((10, 10), dtype=int)
    for g, p in zip(gold_labels, pred_labels):
        cm[g, p] += 1

    print(f"\nAccuracy: {accuracy:.4%}  ({correct}/{len(pred_labels)})\n")
    print("Confusion matrix (rows = golden, cols = predicted):")
    for r in range(10):
        print(" ".join(f"{cm[r,c]:5d}" for c in range(10)))

if __name__ == "__main__":
    main()
