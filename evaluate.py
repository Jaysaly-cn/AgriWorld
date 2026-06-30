"""Compatibility entry point for `python evaluate.py`."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.evaluate", run_name="__main__")
