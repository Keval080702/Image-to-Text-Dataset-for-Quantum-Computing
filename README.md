# Quantum Circuit Dataset & Collection Pipeline

A robust, multi-stage machine learning pipeline designed to harvest high-quality quantum circuit diagrams from arXiv papers. This system processes PDFs, extracts figures, and filters them using a hybrid approach of Text Analysis, OCR, and Vision-Language Models (CLIP).

## Key Features
1.  **Multi-Stage Classifier**:
    *   **Stage 0: Geometric Filter**: Rejects images with too few horizontal lines (wires).
    *   **Stage 1a: Text Analysis**: Analyzes captions for circuit keywords ("qubit", "CNOT") and negative signals ("plot", "vs time").
    *   **Stage 1b: Ensemble OCR**: Uses EasyOCR + Tesseract to detect gate tokens (H, CX, RZ) and reject plots (axes, numbers).
    *   **Stage 2: Semantic Vision**: Uses OpenAI's **CLIP** model to compare image embeddings against "Quantum Circuit" vs "Scientific Plot" text prototypes.
2.  **Self-Healing Data Sync**: Automatically ensures consistency between physical image files and the JSON dataset.
3.  **Resume Capability**: Tracks processed papers in a CSV file to allow stopping and resuming without data loss.

## Project Structure
```
.
├── Main_integrated_pipeline.py  # Entry point. Orchestrates download, extraction, and classification.
├── config.py                    # Centralized configuration parameters.
├── dataset_builder.py           # Core logic for processing papers and building the dataset.
├── main_classifier.py           # Classifier implementation (Stage 0, 1a, 1b, and 2).
├── classifiers/                 # Module containing specific stage implementations:
│   ├── stage0_geometric.py      # Geometric analysis (OpenCV).
│   ├── stage1_text.py           # Text/Caption analysis.
│   ├── stage1_ocr.py            # OCR analysis (EasyOCR/Tesseract).
│   ├── stage2_embedding.py      # Vision analysis (CLIP).
│   └── fusion_classifier.py     # Logic to combine signals from all stages.
├── metadata_generator.py        # Generates QASM-like metadata for accepted circuits.
├── monitoring.py                # Logging system (saves accepted/rejected images to folders).
├── pdf_processor.py             # Helper utility for PDF operations.
├── req.txt                      # Dependency list.
├── quantum_circuits_8/          # OUTPUT: Contains the final dataset (generated).
│   ├── images_8/                # Collected circuit images.
│   └── dataset_8.json           # Metadata (arXiv ID, caption, gate counts).
└── paper_list_counts_8.csv      # Log of all processed papers (used for resuming).
```

## Installation

1.  **Prerequisites**: Python 3.8+, CUDA (optional but recommended for CLIP/OCR).
2.  **Install Dependencies**:
    ```bash
    pip install -r req.txt
    ```
    *Note: If `easyocr` fails, you may need to install PyTorch separately first.*

## Usage

### 1. Run the Pipeline
To start (or resume) the collection process:
```bash
python Main_integrated_pipeline.py
```
*   **Input**: `paper_list_8.txt` (List of arXiv IDs - generated or expected in root).
*   **Logs**: Console output and `paper_list_counts_8.csv`.

### 2. Monitor Progress
Check the CSV log to see real-time progress. The pipeline also creates a `stage_monitoring` directory to visualize accepted/rejected images at each stage.

## Methodology Details

### The "Fusion" Logic
The pipeline uses a strict hierarchy to minimize False Positives:
1.  **Stage 0 (Lines)**: Must have horizontal wires (Physics-inspired heuristic).
2.  **Stage 1a (Text)**: Reject if caption explicitly says "Plot" or "Function of".
3.  **Stage 1b (OCR)**: Reject if image contains axes/ticks/numbers.
4.  **Reference Check**: Does it have gate names? (e.g. "Hadamard", "CNOT").
5.  **Stage 2 (Vision)**: If uncertain, trust CLIP's visual similarity score.

### Metadata
The `dataset_8.json` contains:
*   `arxiv_id`: Source paper.
*   `caption`: Extracted figure caption.
*   `quantum_gates`: Detected gates (e.g., `['CNOT', 'H', 'Measure']`).
*   `quantum_problem`: Identified problem domain (e.g. VQE, QAOA).
*   `image_filename`: Filename of the associated image.
