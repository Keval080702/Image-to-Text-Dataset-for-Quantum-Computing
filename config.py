"""
Configuration Module
====================

Centralized configuration parameters and constants for the Quantum Circuit Dataset Builder.
"""

from __future__ import annotations
from pathlib import Path

# ==================== Stage 0: Geometric Filter Config ====================
MIN_HORIZONTAL_WIRES = 2
WIRE_LENGTH_RATIO = 0.20  # Wires must be at least 20% of image width

# ==================== Stage 2: Embedding Thresholds ====================
EMBED_HIGH_ACCEPT = 0.25
EMBED_HIGH_REJECT = -0.05


class Config:
    """
    Configuration parameters for the Quantum Circuit Dataset Builder.

    Attributes
    ----------
    TARGET_CIRCUITS : int
        The goal number of quantum circuits to collect.
    EXAM_ID : str
        Identifier for the current exam/run (e.g., "8").
    BASE_DIR : Path
        Absolute base path of the project.
    PDF_DIR : Path
        Directory to store downloaded arXiv PDFs.
    FIGURES_DIR : Path
        Directory to store extracted figures from PDFs.
    DATASET_DIR : Path
        Root directory for the generated dataset.
    DATASET_IMAGES_DIR : Path
        Sub-directory for the final accepted images.
    PAPER_LIST_FILE : Path
        Path to the text file containing the list of arXiv IDs.
    PAPER_COUNTS_CSV : Path
        Path to the CSV file tracking processing counts per paper.
    DATASET_JSON : Path
        Path to the main dataset JSON file.
    IMAGES_SCALE : float
        Scale factor for rasterizing PDF figures (default: 4.0).
    SLEEP_BETWEEN_PAPERS : int
        Seconds to wait between processing papers to be polite to arXiv.
    USE_ADVANCED_CLASSIFIER : bool
        Flag to enable/disable the advanced multi-stage classifier.
    """
    # Target
    TARGET_CIRCUITS = 250
    EXAM_ID = "8"

    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    PDF_DIR = BASE_DIR / "pdfs"
    FIGURES_DIR = BASE_DIR / "figures_out"
    DATASET_DIR = BASE_DIR / f"quantum_circuits_{EXAM_ID}"
    DATASET_IMAGES_DIR = DATASET_DIR / f"images_{EXAM_ID}"

    # Output files
    PAPER_LIST_FILE = BASE_DIR / "paper_list_8.txt"
    PAPER_COUNTS_CSV = BASE_DIR / f"paper_list_counts_{EXAM_ID}.csv"
    DATASET_JSON = DATASET_DIR / f"dataset_{EXAM_ID}.json"
    ALL_CLASSIFICATIONS_JSON = DATASET_DIR / f"all_classifications_{EXAM_ID}.json"
    AUDIT_LOG_JSON = DATASET_DIR / f"audit_log_{EXAM_ID}.json"

    # Processing
    IMAGES_SCALE = 4.0  # safer default; can be overridden by env DOC_IMAGES_SCALE
    SLEEP_BETWEEN_PAPERS = 5
    USE_ADVANCED_CLASSIFIER = True
