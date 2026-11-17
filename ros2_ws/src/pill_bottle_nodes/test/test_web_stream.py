# The live camera link (no robot): python3 -m pytest test/test_web_stream.py
import urllib.request

import numpy as np

from pill_bottle_nodes.web_stream import WebStream


def test_page_snapshot_and_stream():
    s = WebStream(18090, "test")
    try:
        other = WebStream(18090, "second")          # port taken -> the next one
        assert other.port != s.port
        other.close()
        assert s.url.startswith("http://") and s.url.endswith(f":{s.port}")
        base = f"http://127.0.0.1:{s.port}"
        assert b'<img src="/stream">' in urllib.request.urlopen(base + "/", timeout=5).read()
        img = np.zeros((40, 30, 3), np.uint8)
        img[:, :, 2] = 255
        s.update(img)
        jpg = urllib.request.urlopen(base + "/snapshot.jpg", timeout=5).read()
        assert jpg[:2] == b"\xff\xd8"
        r = urllib.request.urlopen(base + "/stream", timeout=5)
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        assert b"--frame" in r.read(200)
        r.close()
    finally:
        s.close()
