"""
Generate an hls4ml project for either model (baseline or quantised).

cfg['hls'] example:
  backend:   Vitis         # or Vivado
  clock:     5             # ns
  reuse:     32
  strategy:  Latency
"""

from pathlib import Path
import hls4ml
from ai_on_fpga.utils import log


def _config_for(model, cfg: dict, backend: str):
    hcfg = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        backend=backend,
    )
    hcfg["Model"]["Precision"] = "ap_fixed<16,6>"
    for lname, lcfg in hcfg["LayerName"].items():
        lcfg["Strategy"] = cfg.get("strategy", "Latency")
        lcfg["ReuseFactor"] = cfg.get("reuse", 1)
    return hcfg


def create(
    model,
    board_spec: dict,
    cfg: dict,
    out_dir: Path,
    tag: str = "baseline",
):
    """
    Converts the given Keras model into an HLS project directory and *returns*
    the `hls4ml.model.hls_model.HLSModel` instance so the caller can run build().
    """
    prj = out_dir / f"hls_{tag}"
    backend = cfg.get("backend", "Vitis")
    hls_cfg = _config_for(model, cfg, backend)

    log.info("⏳  Converting Keras → HLS (%s backend)", backend)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_cfg,
        backend=backend,
        output_dir=prj.as_posix(),
        part=board_spec["part"],
        io_type="io_stream",
        clock_period=cfg.get("clock", 5),
    )
    hls_model.compile()
    log.info("✅  HLS project generated → %s", prj)
    return hls_model
