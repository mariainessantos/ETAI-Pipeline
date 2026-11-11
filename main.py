
"""
COMPAS pipeline entry point.

Run with:
    python main.py
"""

from src.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline("config.yaml")
