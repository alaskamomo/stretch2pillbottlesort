"""
web_stream.py - show a camera picture in any browser (Mac, phone) on the same network. No ROS.

  stream = WebStream(8080, "T2 head camera")
  print(stream.url)          # http://10.0.11.20:8080
  stream.update(bgr_image)   # call with every new picture

The page refreshes itself (MJPEG). /snapshot.jpg is the newest still picture.
If the port is taken (another tester still running) the next free one is used.
The same file is in bottle_tracking/web_stream.py (the two packages don't import each other).
"""

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width">
<title>{title}</title><style>body{{margin:0;background:#111;color:#ddd;font:14px sans-serif;text-align:center}}
img{{max-width:100%;max-height:92vh}}</style></head>
<body><div style="padding:6px">{title}</div><img src="/stream"></body></html>"""


def robot_ip():
    """The address other computers use to reach this one (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        s.close()


class WebStream:
    def __init__(self, port=8080, title="Robot camera", tries=10, fps=10.0):
        self.title, self.period = title, 1.0 / fps
        self._jpg, self._n = None, 0
        self._lock = threading.Lock()
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/stream"):
                    return stream._serve_stream(self)
                if self.path.startswith("/snapshot"):
                    jpg = stream.jpeg()
                    if jpg is None:
                        self.send_error(503, "no picture yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                body = PAGE.format(title=stream.title).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server, err = None, None
        for p in range(int(port), int(port) + tries):
            try:
                self.server = ThreadingHTTPServer(("0.0.0.0", p), Handler)
                break
            except OSError as e:
                err = e
        if self.server is None:
            raise RuntimeError(f"no free port {port}-{port + tries - 1}: {err}")
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://{robot_ip()}:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, bgr):
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpg, self._n = buf.tobytes(), self._n + 1

    def jpeg(self):
        with self._lock:
            return self._jpg

    def _serve_stream(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()
        last = -1
        try:
            while True:
                with self._lock:
                    jpg, n = self._jpg, self._n
                if jpg is not None and n != last:
                    h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                  + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    last = n
                time.sleep(self.period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
