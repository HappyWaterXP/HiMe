#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self._range = None
        path = self.translate_path(self.path)

        if os.path.isdir(path):
            return super().send_head()

        ctype = self.guess_type(path)
        try:
            file_handle = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(file_handle.fileno()).st_size
        range_header = self.headers.get("Range")
        match = re.match(r"bytes=(\d*)-(\d*)$", range_header or "")

        if match:
            start_text, end_text = match.groups()
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
            end = min(end, size - 1)

            if start > end or start >= size:
                file_handle.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None

            self._range = (start, end)
            file_handle.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            return file_handle

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        return file_handle

    def copyfile(self, source, outputfile):
        try:
            if not self._range:
                shutil.copyfileobj(source, outputfile)
                return

            start, end = self._range
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Serve the HiMe demo with video seeking support.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("", args.port), RangeRequestHandler)
    print(f"Serving on http://localhost:{args.port}/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
