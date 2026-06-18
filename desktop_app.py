"""Desktop application — launches RAG system in a native Windows window."""

import os
import sys
import time
import subprocess
from pathlib import Path

if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def find_free_port(start=8501):
    import socket
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 8501


def wait_for_server(url, timeout=60):
    """Wait until the server responds, with detailed logging."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # Kill any existing streamlit on our port
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if s.connect_ex(("127.0.0.1", port)) == 0:
        s.close()
        port = find_free_port(port + 1)
        url = f"http://127.0.0.1:{port}"

    # Launch Streamlit subprocess
    env = os.environ.copy()

    server_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "web" / "app.py"),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.enableWebsocketCompression", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # Wait patiently
    print(f"Waiting for server on {url} ...")
    if not wait_for_server(url, timeout=60):
        print("Server failed to start!")
        server_process.terminate()
        sys.exit(1)

    print("Server ready! Opening window...")
    # Extra delay to ensure Streamlit is fully initialized
    time.sleep(2)

    # Open native window
    try:
        import webview
        webview.create_window(
            title="RAG 知识库问答系统",
            url=url,
            width=1200,
            height=800,
            min_size=(900, 600),
            resizable=True,
        )
        webview.start(debug=False)
    finally:
        print("Shutting down...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("Goodbye!")


if __name__ == "__main__":
    main()
