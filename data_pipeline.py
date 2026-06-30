"""Compatibility entry point for `python data_pipeline.py`."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.data_pipeline", run_name="__main__")
