from pathlib import Path

def getReports(indir, project_name, clk_period_ns: float = 5.0) -> dict:
    indir = Path(indir)
    data = {}

    # Paths
    report_vsynth = indir / "vivado_synth.rpt"
    report_csynth = indir / f"{project_name}_prj/sol_synth/syn/report/{project_name}_csynth.rpt"

    if not (report_vsynth.is_file() and report_csynth.is_file()):
        print(f"[WARN] Missing reports under {indir}")
        return data

    print(f"[INFO] Found vsynth + csynth in {indir}, parsing...")

    # --- Parse vsynth resources ---
    def extract_field(lines, key, val_idx=2, rel_idx=5, cast=int):
        """Helper to extract utilization and relative usage."""
        for line in lines:
            if key in line:
                parts = [p.strip() for p in line.split("|")]
                try:
                    used = cast(parts[val_idx])
                    rel  = float(parts[rel_idx])
                except (ValueError, IndexError):
                    used, rel = None, None
                return used, rel
        return None, None

    with report_vsynth.open() as f:
        lines = f.readlines()

    lut, lut_rel       = extract_field(lines, "CLB LUTs*", cast=int)
    ff, ff_rel         = extract_field(lines, "CLB Registers", cast=int)
    bram, bram_rel     = extract_field(lines, "Block RAM Tile", cast=float)
    dsp, dsp_rel       = extract_field(lines, "DSPs", cast=int)

    data.update({
        "lut": lut, "lut_rel": lut_rel,
        "ff": ff,   "ff_rel": ff_rel,
        "bram": bram, "bram_rel": bram_rel,
        "dsp": dsp, "dsp_rel": dsp_rel,
    })

    # --- Parse csynth latency ---
    with report_csynth.open() as f:
        lines = f.readlines()

    lat_idx = next(
        (i for i, line in enumerate(lines) if "Latency (cycles)" in line), None
    )
    if lat_idx is not None and lat_idx + 3 < len(lines):
        parts = [p.strip() for p in lines[lat_idx + 3].split("|")]
        try:
            latency_cycles = int(parts[2])
            ii             = int(parts[6])
        except (ValueError, IndexError):
            latency_cycles, ii = None, None
        latency_us = latency_cycles * clk_period_ns / 1000.0 if latency_cycles else None

        data.update({
            "latency_clks": latency_cycles,
            "latency_us": latency_us,
            "latency_ii": ii,
        })

    return data
