"""Compatibility entry point for `python pretrain.py`."""

from runpy import run_module


if __name__ == "__main__":
    run_module("agriworld.pretrain", run_name="__main__")
