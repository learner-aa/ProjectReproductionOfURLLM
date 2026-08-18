import http.server
import os
import socketserver

DIRECTORY = os.path.join(os.path.dirname(__file__), "dist")

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_GET(self):
        if "." not in os.path.basename(self.path):
            self.path = "/index.html"
        return super().do_GET()

class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    port = 6006
    with ReuseTCPServer(("0.0.0.0", port), NoCacheHandler) as httpd:
        print(f"Serving on port {port}")
        httpd.serve_forever()
