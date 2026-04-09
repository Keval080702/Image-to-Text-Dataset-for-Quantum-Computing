"""
Main Entry Point
================

Entry point for the Quantum Circuit Dataset Builder.
"""

import sys
import os
from pathlib import Path

# Ensure imports work correctly
sys.path.append(str(Path(__file__).resolve().parent))

# Set up stage monitoring directory
os.environ["STAGE_MONITOR_DIR"] = str(Path(__file__).resolve().parent / "stage_monitoring")
print(f"[INFO] STAGE_MONITOR_DIR = {os.environ['STAGE_MONITOR_DIR']}")

from config import Config
from dataset_builder import QuantumCircuitDatasetBuilder


def main():
    """Main entry point."""
    config = Config()
    builder = QuantumCircuitDatasetBuilder(config)
    builder.build_dataset()


if __name__ == "__main__":
    main()
