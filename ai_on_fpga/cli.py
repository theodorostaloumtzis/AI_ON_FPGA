"""
Command-line front-end:

$ aionfpga build --config mynet.yaml --board pynq-z2 --out build/
"""
from pathlib import Path
import typer
from ai_on_fpga.core.pipeline import build_pipeline

app = typer.Typer(help="Train → quantise → HLS → bitstream in one go")

@app.command()
def build(
    config: Path = typer.Option(..., help="YAML describing data/model/train"),
    board: str = typer.Option("pynq-z2", help="Target board key (see boards/)"),
    out:   Path = typer.Option(Path("./build"), help="Output directory"),
):
    """Run the full flow."""
    build_pipeline(config, board, out)

if __name__ == "__main__":
    app()
