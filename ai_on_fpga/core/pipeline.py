"""
High-level façade that the CLI calls.
"""
from pathlib import Path
import yaml

from ai_on_fpga.utils import log, timer
from . import data, models, train, quantize, hls, bitstream

def _load_yaml(p: Path):
    with open(p) as f:
        return yaml.safe_load(f)

def build_pipeline(cfg_file: Path, board_key: str, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg   = _load_yaml(cfg_file)
    board = _load_yaml(Path(__file__).parent.parent / "boards" / f"{board_key}.yaml")

    with timer("total pipeline"):
        # 1. data
        ds_train, ds_val = data.prepare(cfg.get("data", {}))

        # 2. model & training
        model = models.build(cfg.get("model", {}))
        model = train.fit(model, ds_train, ds_val, cfg.get("train", {}), out_dir)

        # 3. pruning / quantisation
    baseline, qmodel = quantize.apply(
        model, cfg.get("quantize", {}), out_dir, ds_train, ds_val
    )

    # 4. HLS (baseline)
    hls_baseline = hls.create(baseline, board, cfg.get("hls", {}), out_dir, tag="baseline")
    bit_baseline = bitstream.build(hls_baseline, cfg.get("impl", {}), out_dir)

    # 5. HLS (quantised) – optional
    if qmodel is not None:
        hls_quant = hls.create(qmodel, board, cfg.get("hls", {}), out_dir, tag="quant")
        bit_quant = bitstream.build(hls_quant, cfg.get("impl", {}), out_dir)

    log.info("Pipeline finished → %s", bit_quant)
