"""
Data Models Module
==================

Data structures and models used across the Quantum Circuit Dataset Builder.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class QuantumCircuitEntry:
    """
    Data model representing a single accepted quantum circuit image.

    Attributes
    ----------
    image_filename : str
        The filename of the saved image (e.g., "arxivID_p01_f01.png").
    arxiv_id : str
        The arXiv ID of the source paper.
    page_number : int
        The page number where the figure was found.
    figure_number : int
        The figure number extracted from docling (may be negative/internal).
    quantum_gates : List[str]
        List of detected quantum gates (e.g., "H", "CNOT").
    quantum_problem : str
        Identified problem domain (e.g., "VQE", "QOA", "unknown").
    descriptions : List[str]
        Captions associated with the figure.
    text_positions : List[Tuple[int, int]]
        Character offsets of the caption text in the full text (if available).
    """
    image_filename: str
    arxiv_id: str
    page_number: int
    figure_number: int
    quantum_gates: List[str]
    quantum_problem: str
    descriptions: List[str]
    text_positions: List[Tuple[int, int]]


@dataclass
class PaperProcessingResult:
    """
    Summary of the processing results for a single arXiv paper.

    Attributes
    ----------
    arxiv_id : str
        The arXiv ID of the paper.
    pdf_path : Optional[str]
        Path to the downloaded PDF, or None if download failed.
    status : str
        Outcome status (e.g., "SUCCESS", "FAILED").
    total_figures_extracted : int
        Total number of figures extracted from the PDF.
    quantum_circuits_found : int
        Number of figures classified as quantum circuits.
    error_message : str, optional
        Analysis of the error if status is "FAILED".
    """
    arxiv_id: str
    pdf_path: Optional[str]
    status: str
    total_figures_extracted: int
    quantum_circuits_found: int
    error_message: str = ""


@dataclass
class VisionAnalysis:
    """
    Result of the VLM/Embedding visual analysis.

    Attributes
    ----------
    scores : Dict[str, int]
        Raw similarity scores (e.g., sim_img_circuit).
    detected_gate_text : List[str]
        (Unused in current EmbeddingJudge) Placeholders for future VLM OCR.
    visual_summary : str
        Human-readable summary of the visual analysis.
    """
    scores: Dict[str, int]
    detected_gate_text: List[str]
    visual_summary: str


@dataclass
class ClassificationResult:
    """
    Final decision object from the Fusion Classifier.

    Attributes
    ----------
    decision : str
        "ACCEPT" or "REJECT".
    confidence_tag : str
        "HIGH", "MEDIUM", "LOW" (or corresponding rejection tags).
    reason : str
        Explanation for the decision.
    evidence : Dict[str, Any]
        Aggregated signals from all stages.
    quantum_gates : List[str]
        Detected gates (merged from multiple stages).
    quantum_problem : str
        Identified problem domain.
    descriptions : List[str]
        Captions/Descriptions used.
    text_positions : List[Tuple[int, int]]
        Token positions of descriptions.
    """
    decision: str                    # ACCEPT / REJECT
    confidence_tag: str              # HIGH / MEDIUM / LOW
    reason: str
    evidence: Dict[str, Any]

    quantum_gates: List[str] = field(default_factory=list)
    quantum_problem: str = ""
    descriptions: List[str] = field(default_factory=list)
    text_positions: List[Tuple[int, int]] = field(default_factory=list)
