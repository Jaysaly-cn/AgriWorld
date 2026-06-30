"""Compatibility entry point for `python audit_data.py`."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.audit_data", run_name="__main__")
