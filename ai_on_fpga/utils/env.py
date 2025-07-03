# utils/env.py
import os, subprocess
from pathlib import Path
from .log import log

def load_xilinx_env(tool_root: str):
    """
    Reads the 'settings64.sh' produced by Xilinx and returns a dict
    of environment vars we can inject into subprocesses.
    """
    settings = Path(tool_root) / "settings64.sh"
    if not settings.exists():
        raise FileNotFoundError(settings)
    # Run the script in a clean shell and print the env as NUL-separated list
    cmd = ["bash", "-c", f"source {settings} && env -0"]
    out = subprocess.check_output(cmd)
    env = {
        k: v
        for k, v in (
            line.decode().split("=", 1)
            for line in out.split(b"\0")
            if b"=" in line
        )
    }
    log.info("✔ Loaded Xilinx env from %s", settings)
    return env
