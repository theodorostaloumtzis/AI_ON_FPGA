"""
Call hls_model.build() which in turn invokes Vivado/Vitis HLS synthesis.

cfg['impl'] example:
  synth:      true     # run C-synth
  export:     true     # generate IP / bitstream
  csim:       false    # skip C-sim
"""

from pathlib import Path
from ai_on_fpga.utils import log


def build(hls_model, cfg: dict, out_dir: Path) -> Path:
    opts = dict(
        csim=cfg.get("csim", False),
        synth=cfg.get("synth", True),
        export=cfg.get("export", True),
    )
    log.info("⏳  Launching HLS synthesis (%s)", ", ".join(k for k, v in opts.items() if v))
    hls_model.build(**opts)

    # Vitis places the bitstream in <proj>/myproject/solution/syn/vivado/*.bit
    prj_dir = Path(hls_model.config.get_output_dir())
    bit_files = list(prj_dir.glob("**/*.bit"))

    if not bit_files:
        raise FileNotFoundError("No .bit file produced – check synthesis log!")
    bit_path = bit_files[0]
    log.info("✅  Bitstream ready → %s", bit_path)
    return bit_path
