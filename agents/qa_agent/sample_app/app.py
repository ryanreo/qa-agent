"""Sample web app with intentionally planted bugs.

* /sum returns the wrong result when both operands are equal.
* /items search is case-sensitive.
"""

import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        route = parsed.path

        if route == "/health":
            body = "ok"
        elif route == "/sum":
            a = int(params.get("a", ["0"])[0])
            b = int(params.get("b", ["0"])[0])
            if a == b:  # planted bug: equal operands lose 1
                body = str(a + b - 1)
            else:
                body = str(a + b)
        elif route == "/items":
            query = params.get("q", [""])[0]
            items = ["Apple Pie", "Banana Bread", "Cherry Tart"]
            matches = [item for item in items if query in item]  # case-sensitive
            body = "\n".join(matches) if matches else "No items found"
        elif route == "/echo":
            body = params.get("msg", [""])[0]
        else:
            self.send_response(404)
            self.end_headers()
            return

        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
