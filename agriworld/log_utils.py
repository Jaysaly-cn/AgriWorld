"""Small helpers for teeing console output to result log files."""

import contextlib
import os
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
