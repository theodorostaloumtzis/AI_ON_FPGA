# netron_embed.py

import socket
import threading
import time
from IPython.display import IFrame, display
import netron


def find_free_port():
    """Find a free port for the Netron server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_netron_server(model_path, port):
    """Start Netron server using the best available method."""
    try:
        netron.start(model_path, port=port, browse=False)
    except TypeError:
        try:
            netron.serve(model_path, port=port, browse=False)
        except TypeError:
            import subprocess
            subprocess.Popen(['netron', model_path, f'--port={port}'])


def view_model(model_path, width='100%', height=600):
    """
    Launch Netron viewer in a background thread and display in notebook.
    
    Parameters:
        model_path (str): Path to model file (.onnx, .h5, .pb, etc.)
        width (str|int): IFrame width
        height (str|int): IFrame height
    """
    port = find_free_port()

    def run():
        start_netron_server(model_path, port)

    threading.Thread(target=run, daemon=True).start()
    time.sleep(2)  # allow time for Netron to start

    display(IFrame(src=f"http://localhost:{port}", width=width, height=height))
