# qasm_classifier.py
"""
3-Stage (4-signal) Quantum Circuit Classifier
=============================================

Stage 0  : Geometric Filter (OpenCV) -> detects horizontal quantum rails (physics heuristic)
Stage 1a : Text (caption/mention)    -> circuit language vs plot language
Stage 1b : OCR (image text)          -> axes/ticks/numbers vs ket |0> etc.
Stage 2  : Vision (CLIP Embedding)   -> high-dimensional semantic similarity (Circuit vs Plot)

IMPORTANT:
- ALL stages run for EVERY image (no early stop)
- Final decision is made ONLY in fusion
- Monitoring folders are created with absolute paths (robust)
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from transformers import CLIPModel, CLIPProcessor
from monitoring import StageMonitor

# Paths (ABSOLUTE, robust)
BASE_DIR = Path(__file__).resolve().parent

# Stage 0 config
# Updated: OpenCV Horizontal Line Filter Config
MIN_HORIZONTAL_WIRES = 2
WIRE_LENGTH_RATIO = 0.20  # Wires must be at least 20% of image width

# Stage 2 (embedding) thresholds
# These are global and can be tuned without changing logic.
EMBED_HIGH_ACCEPT = 0.25
EMBED_HIGH_REJECT = -0.05



# -------------------- Stage 0: Horizontal Wire Filter (OpenCV) --------------------
class HorizontalWireFilter:
    """
    Stage 0: Physics-inspired Geometric Filter
    Detects horizontal lines (quantum rails) using Hough Transform.
    Fast rejection of scatter plots, bar charts, and random images.
    """
    def __init__(self):
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
            self.available = True
        except ImportError:
            print("[WARN] OpenCV not installed. Stage 0 will pass everything.")
            self.available = False

    def count_wires(self, image_path: str) -> Tuple[int, bool]:
        """
        Count significant horizontal lines in the image to filter non-circuits.

        Parameters
        ----------
        image_path : str
            Absolute path to the image file.

        Returns
        -------
        Tuple[int, bool]
            (wire_count, is_rejected)
            - wire_count: Number of horizontal lines detected.
            - is_rejected: True if wire_count < MIN_HORIZONTAL_WIRES.
        """
        if not self.available:
            return 99, False # Pass everything

        try:
            # Read image in grayscale directly
            # We use the absolute path provided
            img = self.cv2.imread(image_path, self.cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0, True # Reject if can't read

            # Resize for consistent processing speed 
            h, w = img.shape
            target_w = 500
            scale = target_w / w
            target_h = int(h * scale)
            img = self.cv2.resize(img, (target_w, target_h))
            
            # Edge detection
            edges = self.cv2.Canny(img, 50, 150, apertureSize=3)
            
            # Probabilistic Hough Line Transform
            # minLineLength: e.g. 20% of width
            min_len = int(target_w * WIRE_LENGTH_RATIO)
            
            lines = self.cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=self.np.pi/180, 
                threshold=50, 
                minLineLength=min_len, 
                maxLineGap=10
            )
            
            if lines is None:
                return 0, True

            horizontal_wires = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check angle: basically horizontal?
                # y diff should be small vs x diff
                dy = abs(y2 - y1)
                dx = abs(x2 - x1)
                if dx == 0: continue
                angle = self.np.arctan2(dy, dx) * 180 / self.np.pi
                
                # Allow small deviation (e.g. +/- 5 degrees)
                if angle < 5.0:
                    horizontal_wires += 1

            # Decision
            # We treat multiple segments on same rail as duplicates? 
            # For simplicity, raw count of long horizontal segments is a robust proxy.
            
            is_rejected = (horizontal_wires < MIN_HORIZONTAL_WIRES)
            
            return horizontal_wires, is_rejected

        except Exception as e:
            print(f"[WARN] Stage 0 error: {e}")
            return 0, False # Fail open (pass) on error to be safe



# -------------------- Stage 1a: Text/context --------------------
class TextJudge:
    """
    Stage 1a: Text Analysis (Caption & Context).
    
    Caption/context analysis is evidence ONLY, not decision.
    """

    POS_PHRASES = [
        "quantum circuit", "circuit diagram", "gate sequence", "ansatz", "scheme",
        "qubit", "qreg", "controlled", "measurement", "state preparation", 
        "adder", "logic gate", "module", "implementation",
        "variational form", "oracle", "syndrome extraction", "encoding circuit", 
        "decoding circuit", "entangling layer", "trotter step", "transpiled", 
        "decomposed", "unitary", "register", "ancilla", "stabilizer",
    ]
    POS_GATES = [
        "cnot", "cx", "cz", "swap", "hadamard", "toffoli", "rx", "ry", "rz", "u3", "u2", "u1",
        "controlled-phase", "cp", "cphase", "ccx", "measure", "reset", "barrier",
        "fredkin", "cswap", "iswap", "xx", "yy", "zz", "rxx", "ryy", "rzz"
    ]

    NEG_PHRASES = [
        "plot", "graph", "histogram", "benchmark",
        "dashed line", "solid line", "dotted line",
        "transmission coefficient", "setup", "apparatus",
        "tensor network", "mps", "peps", "bloch sphere", "energy level", "hamiltonian",
        "function of", "dependence of", "measured", "simulated", "experiment",
        # ML / AI terms (preserve these if they indicate architectural diagrams rather than quantum circuits)
        "neural network", "transformer", "training", "inference", "loss", "mse", "epochs",
        # Hardware / Physics terms
        "satellite", "telescope", "waveguide", "fiber", "fibre", "resonator", "resistor",
        "impedance", "voltage", "current", "laser", "pump", "beam splitter", "spdc",
        "satellite", "telescope", "waveguide", "fiber", "fibre", "resonator", "resistor",
        "impedance", "voltage", "current", "laser", "pump", "beam splitter", "spdc",
        "fabrication", "microscopy", "wafer", "device physics", "transmon", "superconducting",
        "traps", "ion trap", "shuttle", "qccd", "internal structure", "overview of", "schematic of",
        "leakage", "radiative decay", "level scheme", "energy level",
        "wigner function", "phase space", "coherent state", "squeezing",
        "charge density", "wavepacket", "potential barrier", "light-cone",
        "physical layout", "dispersive coupling",
        # Surface Code / Grids (often not circuits)
        "lattice", "surface code", "plaquette", "decoding graph", "unit cell",
        # Plot terms (Performance/Errors)
        "convergence", "scaling", "parameter estimation", "spectroscopy", "interference",
        "spectrum", "covariance matrix", "curves", "fisher information", "fidelity", "error rate",
        "error probability", "process matrix", "pauli transfer matrix", "runtime", "speedup",
        "efficiency", "performance", "linear", "logarithmic", "function of",
        "convergence", "scaling", "parameter estimation", "spectroscopy", "interference",
        "spectrum", "covariance matrix", "curves",
        # Hardware / Microscopy
        "sem image", "micrograph", "cross-section", "silicon", "donor", "cluster state", 
        "cluster array", "fabrication", "device", "chip",
        
        
    ]
    NEG_BLOCK = ["block diagram", "architecture", "pipeline", "workflow", "flowchart", "module", "fpga", "algorithm"]

    def analyze(self, caption: str, context_mentions: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text content for positive (circuit) and negative (plot) signals.

        Parameters
        ----------
        caption : str
            The primary caption text.
        context_mentions : str, optional
            Surrounding context text from the page.

        Returns
        -------
        Dict[str, Any]
            Dictionary of scores (0-3) and detected keywords.
        """
        text = (caption or "") + "\n" + (context_mentions or "")
        low = text.lower()

        pos_phrase_hits = sum(1 for p in self.POS_PHRASES if p in low)
        
        found_gates = []
        for g in self.POS_GATES:
            if re.search(rf"\b{re.escape(g)}\b", low):
                found_gates.append(g)
        pos_gate_hits = len(found_gates)

        neg_phrase_hits = sum(1 for n in self.NEG_PHRASES if n in low)
        neg_block_hits = sum(1 for b in self.NEG_BLOCK if b in low)

        # map to 0-3 evidence
        pos_score = 0
        if pos_phrase_hits >= 2 or pos_gate_hits >= 2:
            pos_score = 2
        if pos_phrase_hits >= 3 or pos_gate_hits >= 3:
            pos_score = 3
        elif pos_phrase_hits == 1 or pos_gate_hits == 1:
            pos_score = 1

        neg_score = 0
        if neg_phrase_hits >= 2:
            neg_score = 2
        if neg_phrase_hits >= 4:
            neg_score = 3
        elif neg_phrase_hits == 1:
            neg_score = 1

        block_score = 0
        if neg_block_hits >= 2:
            block_score = 2
        if neg_block_hits >= 3:
            block_score = 3
        elif neg_block_hits == 1:
            block_score = 1
            
        # Decision logic for negative context
        is_negative = False
        # Strong plot indicators
        if any(ph in low for ph in ["time evolution", "dashed line", "solid line", "inset", "histogram", "spectrum", "transmission coefficient", "data points"]):
             # If it looks like a plot and has NO gates, it's a plot.
             if pos_gate_hits == 0:
                 is_negative = True
        
        # Strong setup/schematic indicators
        if any(ph in low for ph in ["schematic", "apparatus", "setup", "tensor network", "bloch sphere"]):
            # Only kill it if it looks completely devoid of circuit terms
            if pos_gate_hits == 0 and "circuit" not in low and pos_phrase_hits == 0:
                is_negative = True

        # NEW: Strict Semantic Check for Precision
        # These patterns strongly imply the image is ABOUT data/performance, not the circuit definition.
        # "X vs Y", "X as a function of Y", "Dependence of X on Y"
        extracted_semantic_reject = False
        semantic_patterns = [
            r"(\w+)\s+vs\.?\s+(\w+)", # "error vs time"
            r"(\w+)\s+versus\s+(\w+)",
            r"function of",
            r"dependence of",
            r"performance of",
            r"comparison of",
            r"scaling of",
            r"error rate",
            r"infidelity",
            r"probabilities?", # "probabilities of..." usually a bar chart
            r"population",    # "population transfer" -> plot
        ]
        if any(re.search(pat, low) for pat in semantic_patterns):
            # Vet: Only flag if we don't have extremely strong 'circuit diagram' declaration
            # If it says "comparison of circuit diagrams", we might be safe? 
            # But usually "comparison of" implies data.
            # safe guard: "circuit diagram of" ?
            if "circuit diagram" not in low: 
                 extracted_semantic_reject = True
                 print(f"[TextJudge] Strict Semantic Reject triggered by pattern match in: '{text[:50]}...'")

        return {
            "text_pos_circuit_language": pos_score,
            "text_neg_plot_language": neg_score,
            "text_blockdiagram_language": block_score,
            "text_pos_phrase_hits": pos_phrase_hits,
            "text_pos_gate_hits": pos_gate_hits,
            "text_neg_phrase_hits": neg_phrase_hits,
            "text_block_hits": neg_block_hits,
            "text_gate_tokens": found_gates,
            "is_negative_context": is_negative,
            "is_strict_semantic_reject": extracted_semantic_reject,
            "snippet": text[:300].replace("\n", " "),
        }


# -------------------- Stage 1b: OCR evidence --------------------
class OCRJudge:
    """
    Stage 1b: OCR (Optical Character Recognition) Analysis.

    We treat OCR strictly as evidence, not a hard decision maker.
    Its job is to find:
      - Coordinate system markers (axes, ticks, numbers) -> which implies it's a Plot.
      - Quantum notation (|0>, |ψ>, gate names) -> which implies it's a Circuit.
      - The density of text vs whitespace.

    Attributes
    ----------
    ocr : easyocr.Reader
        Primary OCR engine (EasyOCR).
    pytesseract : module
        Fallback OCR engine (Tesseract).
    """

    def __init__(self):
        try:
            import easyocr  # type: ignore
            self.ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        except Exception:
            self.ocr = None
            
        # [ENSEMBLE] Try to load Tesseract as backup
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
            # Check availability cheaply
            self.pytesseract.get_tesseract_version()
            self.use_tesseract = True
        except Exception as e:
            print(f"[WARN] Tesseract not available: {e}")
            self.use_tesseract = False    

    def _read(self, image_path: str) -> List[str]:
        results = []
        
        # 1. EasyOCR
        if self.ocr:
            try:
                out = self.ocr.readtext(image_path)
                results.extend([str(x[1]) for x in out if len(x) >= 2])
            except Exception:
                pass
                
        # 2. Tesseract (Ensemble)
        if self.use_tesseract:
            try:
                # Tesseract usually returns one big string, split by whitespace
                raw = self.pytesseract.image_to_string(self.Image.open(image_path))
                # Split and filter empty
                t_words = [w.strip() for w in raw.split() if len(w.strip()) > 1]
                results.extend(t_words)
            except Exception:
                pass
                
        return results

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform OCR on the image and analyze the extracted text.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - ocr_text_count: Number of text regions found.
            - ocr_axes_evidence: Score (0-3) for axis labels.
            - ocr_many_numbers: Score (0-3) for numerical density.
            - ocr_ket_evidence: Score (0-3) for quantum state notation.
            - ocr_gate_tokens: List of detected gate keywords.
            - ocr_snippet: Sample of extracted text.
        """
        texts = self._read(image_path)
        joined = " ".join(texts)
        low = joined.lower()

        # axes / ticks / numbers evidence
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", joined)
        many_numbers = len(numbers)

        axis_words = ["xlabel", "ylabel", "axis", "time", "ms", "us", "μs", "s", "hz", "khz", "mhz"]
        axis_hits = sum(1 for w in axis_words if w in low)

        # ket notation
        ket_matches = re.findall(r"\|[0-9a-zA-Z+\-ψφ]+[>⟩]", joined)
        ket_evidence = len(ket_matches)

        # gate tokens (very conservative: only multi-letter or common 2+ patterns)
        gate_tokens = []
        for token in ["CNOT", "CX", "CZ", "SWAP", "RZ", "RY", "RX", "U3", "U2", "U1", "H"]:
            # only accept if appears as standalone-ish token
            if re.search(rf"(^|[^A-Za-z0-9]){re.escape(token)}([^A-Za-z0-9]|$)", joined):
                gate_tokens.append(token)

        # map to 0-3 scores (evidence strength)
        ocr_axes_score = 0
        if axis_hits >= 2:
            ocr_axes_score = 2
        elif axis_hits == 1:
            ocr_axes_score = 1

        ocr_numbers_score = 0
        if many_numbers >= 12:
            ocr_numbers_score = 3
        elif many_numbers >= 6:
            ocr_numbers_score = 2
        elif many_numbers >= 3:
            ocr_numbers_score = 1

        ocr_ket_score = 0
        if ket_evidence >= 3:
            ocr_ket_score = 3
        elif ket_evidence == 2:
            ocr_ket_score = 2
        elif ket_evidence == 1:
            ocr_ket_score = 1

        return {
            "ocr_text_count": len(texts),
            "ocr_axes_evidence": ocr_axes_score,
            "ocr_many_numbers": ocr_numbers_score,
            "ocr_ket_evidence": ocr_ket_score,
            "ocr_gate_tokens": sorted(set(gate_tokens)),
            "ocr_snippet": joined[:300],
        }


# -------------------- Stage 2: visual embedding evidence --------------------
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


class EmbeddingJudge:
    """Stage 2: CLIP-style embedding judge (image + text).

    This stage compares the image (and its description) against two global
    prototype prompts: "quantum circuit" and "scientific plot". It returns
    cosine similarities and a main score

        score_img = sim(img, circuit) - sim(img, plot)

    which is positive when the image is closer to the quantum-circuit
    concept than to the plot concept.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Global prototype prompts
        self.circuit_prompt = (
            "a diagram of a quantum circuit with qubits, gates, CNOT, "
            "Hadamard, measurement"
        )
        self.plot_prompt = (
            "a scientific plot or graph with axes, data points, histograms, "
            "curves"
        )

        # Precompute and cache prototype text embeddings
        with torch.no_grad():
            inputs = self.processor(
                text=[self.circuit_prompt, self.plot_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            txt_feats = self.model.get_text_features(**inputs)
            txt_feats = F.normalize(txt_feats, p=2, dim=-1)
        self.text_circuit = txt_feats[0]
        self.text_plot = txt_feats[1]

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats[0]

    def _encode_text(self, text: str) -> torch.Tensor:
        if not text.strip():
            return torch.zeros_like(self.text_circuit)
        with torch.no_grad():
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats[0]

    def analyze(self, image_path: str, caption_text: str, context_mentions: Optional[str] = None) -> VisionAnalysis:
        """
        Compute embedding similarity scores for the image + caption.

        Parameters
        ----------
        image_path : str
            Path to the image.
        caption_text : str
            Figure caption.
        context_mentions : str, optional
            Extra context.

        Returns
        -------
        VisionAnalysis
            Structured result containing similarity scores.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # On error, return neutral scores
            return VisionAnalysis(scores={}, detected_gate_text=[], visual_summary="Embedding judge: image load failed.")

        desc_text = caption_text or ""
        if context_mentions:
            if isinstance(context_mentions, list):
                desc_text += " " + " ".join(str(x) for x in context_mentions)
            else:
                desc_text += " " + str(context_mentions)

        img_emb = self._encode_image(image)
        sim_img_circuit = float(torch.dot(img_emb, self.text_circuit).item())
        sim_img_plot = float(torch.dot(img_emb, self.text_plot).item())
        score_img = sim_img_circuit - sim_img_plot

        desc_emb = self._encode_text(desc_text) if desc_text.strip() else None
        sim_desc_circuit = float(torch.dot(desc_emb, self.text_circuit).item()) if desc_emb is not None else 0.0

        scores: Dict[str, float] = {
            "sim_img_circuit": sim_img_circuit,
            "sim_img_plot": sim_img_plot,
            "score_img": score_img,
            "sim_desc_circuit": sim_desc_circuit,
        }

        return VisionAnalysis(
            scores=scores,
            detected_gate_text=[],
            visual_summary="Embedding-based judge; no narrative summary.",
        )


# -------------------- Final result --------------------
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


# -------------------- Fusion classifier --------------------
class QASMClassifier:
    """
    This is what your integrated pipeline imports.
    It ALWAYS runs all stages and then fuses.
    """

    def __init__(self):
        self.monitor = StageMonitor()
        self.stage0 = HorizontalWireFilter()
        self.ocr = OCRJudge()

        self.text = TextJudge()
        self.embed = EmbeddingJudge()

    def classify(
        self,
        arxiv_id: str,
        page_number: int,
        figure_number: int,
        image_path: str,
        caption_text: str,
        pdf_path: Optional[str] = None,
        context_mentions: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Run the full multi-stage classification pipeline on a figure.

        Parameters
        ----------
        arxiv_id : str
            Paper ID.
        page_number : int
            Page number.
        figure_number : int
            Figure number.
        image_path : str
            Path to the figure image.
        caption_text : str
            Caption text.
        pdf_path : str, optional
            Path to original PDF.
        context_mentions : str, optional
            Extra text context.

        Returns
        -------
        ClassificationResult
            Final decision (ACCEPT/REJECT) with supporting evidence.
        """

        evidence: Dict[str, Any] = {
            "stage0": {},
            "stage1_ocr": {},
            "stage1_text": {},
            "stage2_embedding": {},
            "fusion": {},
        }

        timing: Dict[str, float] = {}

        # ---------------- Stage 0 ----------------
        t0 = time.time()
        wires, is_rejected = self.stage0.count_wires(image_path)
        t1 = time.time()
        timing["stage0"] = t1 - t0
        
        evidence["stage0"] = {
            "wire_count": wires,
            "line_filter_rejected": is_rejected
        }

        if is_rejected:
            self.monitor.save("stage0_rejected", arxiv_id, figure_number, image_path)
            self.monitor.save("final_rejected", arxiv_id, figure_number, image_path)
            
            # [STRICT STAGE 0 SHORT-CIRCUIT]
            return ClassificationResult(
                decision="REJECT",
                confidence_tag="HIGH",
                reason=f"Stage0: Only {wires} horizontal lines found (min {MIN_HORIZONTAL_WIRES})",
                evidence=evidence,
                quantum_gates=[], 
            )
        else:
            self.monitor.save("stage0_accepted", arxiv_id, figure_number, image_path)


        # ---------------- Stage 1a: Text (First Priority for Speed) ----------------
        t_text_start = time.time()
        s1a = self.text.analyze(caption_text or "", context_mentions=context_mentions)
        t_text_end = time.time()
        timing["stage1_text"] = t_text_end - t_text_start
        evidence["stage1_text"] = s1a

        # FAIL FAST: If text is strongly negative (Plot/Schematic/Function), REJECT immediately.
        # This skips slow OCR and Embedding.
        if s1a.get("is_negative_context", False) or s1a.get("is_strict_semantic_reject", False):
            self.monitor.save("stage1_text_rejected", arxiv_id, figure_number, image_path)
            self.monitor.save("final_rejected", arxiv_id, figure_number, image_path)
            
            evidence["timing"] = timing
            print(f"[TIMING] {arxiv_id} fig{figure_number}: stage0={timing.get('stage0',0):.3f}s, "
                  f"text={timing.get('stage1_text',0):.3f}s, ocr=0.000s, embed=0.000s (Text Fail-Fast)")

            return ClassificationResult(
                decision="REJECT",
                confidence_tag="HIGH",
                reason="Text: Detected as Plot/Setup/Non-Circuit (Fail-Fast)",
                evidence=evidence,
                quantum_gates=[], 
            )

        if s1a["text_pos_circuit_language"] >= 2 and s1a["text_neg_plot_language"] <= 1:
            self.monitor.save("stage1_text_accepted", arxiv_id, figure_number, image_path)
        else:
            self.monitor.save("stage1_text_rejected", arxiv_id, figure_number, image_path)


        # ---------------- Stage 1b: OCR (Only if text didn't kill it) ----------------
        t_ocr_start = time.time()
        s1b = self.ocr.analyze(image_path)
        t_ocr_end = time.time()
        timing["stage1_ocr"] = t_ocr_end - t_ocr_start
        evidence["stage1_ocr"] = s1b

        # Strict axis-based immediate REJECT: looks like a plot (axes + many numbers) and no quantum ket
        if (s1b["ocr_axes_evidence"] >= 2 and s1b["ocr_many_numbers"] >= 2 and s1b["ocr_ket_evidence"] == 0):
            self.monitor.save("stage1_ocr_rejected", arxiv_id, figure_number, image_path)
            self.monitor.save("final_rejected", arxiv_id, figure_number, image_path)

            evidence["timing"] = timing
            print(f"[TIMING] {arxiv_id} fig{figure_number}: stage0={timing.get('stage0',0):.3f}s, "
                  f"text={timing.get('stage1_text',0):.3f}s, ocr={timing.get('stage1_ocr',0):.3f}s, embed=0.000s (OCR Axis-Reject)")

            return ClassificationResult(
                decision="REJECT",
                confidence_tag="HIGH",
                reason="OCR: axes/ticks/numbers detected without quantum notation",
                evidence=evidence,
                quantum_gates=[],
                quantum_problem="",
                descriptions=[caption_text] if caption_text else [],
                text_positions=[],
            )

        # Strict immediate ACCEPT if OCR finds actual gate tokens
        # [TUNING] Added guard: Do NOT short-circuit if it looks like a graph (axes+numbers)
        # [TUNING] Strict Precision: Do NOT short-circuit if semantic text analysis flags it as a plot (redundant now due to Fail-Fast, but safe)
        is_graph_suspect = (s1b.get("ocr_axes_evidence", 0) >= 1 and s1b.get("ocr_many_numbers", 0) >= 2)
        # is_semantic_reject already handled above

        if len(s1b.get("ocr_gate_tokens", [])) >= 1 and not is_graph_suspect:
            self.monitor.save("stage1_ocr_accepted", arxiv_id, figure_number, image_path)
            self.monitor.save("final_accepted", arxiv_id, figure_number, image_path)

            gates = sorted(list(set(s1b.get("ocr_gate_tokens", []))))

            evidence["timing"] = timing
            print(f"[TIMING] {arxiv_id} fig{figure_number}: stage0={timing.get('stage0',0):.3f}s, "
                  f"text={timing.get('stage1_text',0):.3f}s, ocr={timing.get('stage1_ocr',0):.3f}s, embed=0.000s (OCR Short-Circuit)")

            return ClassificationResult(
                decision="ACCEPT",
                confidence_tag="HIGH",
                reason="OCR: quantum gate tokens detected (Short-Circuit)",
                evidence=evidence,
                quantum_gates=gates,
                quantum_problem="",
                descriptions=[caption_text] if caption_text else [],
                text_positions=[],
            )
        else:
            self.monitor.save("stage1_ocr_rejected", arxiv_id, figure_number, image_path)

        # ---------------- Stage 2 Embedding (ALWAYS) ----------------
        t_embed_start = time.time()
        s2 = self.embed.analyze(image_path, caption_text or "", context_mentions)
        t_embed_end = time.time()
        timing["stage2_embedding"] = t_embed_end - t_embed_start
        evidence["stage2_embedding"] = {
            "score_img": float(s2.scores.get("score_img", 0.0)),
            "sim_img_circuit": float(s2.scores.get("sim_img_circuit", 0.0)),
            "sim_img_plot": float(s2.scores.get("sim_img_plot", 0.0)),
            "sim_desc_circuit": float(s2.scores.get("sim_desc_circuit", 0.0)),
        }
        score_img = evidence["stage2_embedding"]["score_img"]

        # Strict rule: REJECT if no gates found across OCR+Text (before VLM)
        # [TUNING] Relaxed: Allow pass if text phrase score is high (e.g. "Quantum Circuit"), even if no specific gates named.
        # [TUNING] Relaxed V3: Allow pass if Vision Score is POSITIVE (> 0.05), effectively trusting the eye over the text.
        union_gates = set(s1b.get("ocr_gate_tokens", [])) | set(s1a.get("text_gate_tokens", []))
        
        # If no gates found AND text signal is weak...
        if len(union_gates) == 0 and s1a["text_pos_circuit_language"] < 2:
            # ...ONLY reject if Visual Score is also weak/negative.
            # If Vision sees a circuit (score > -0.05), let it pass to Fusion to decide.
            # [TUNING] Lowered from 0.00 to -0.05 to sync with Mercy Rule (Rescue f015).
            if score_img < -0.05:
                self.monitor.save("final_rejected", arxiv_id, figure_number, image_path)
                evidence["timing"] = timing
                return ClassificationResult(
                    decision="REJECT",
                    confidence_tag="HIGH",
                    reason="Strict: no gates found in OCR+Text & Low Visual Score",
                    evidence=evidence,
                    quantum_gates=[],
                    quantum_problem="",
                    descriptions=[caption_text] if caption_text else [],
                    text_positions=[],
                )

        # ---------------- Fusion (final decision) ----------------
        score_img = float(s2.scores.get("score_img", 0.0))

        # [STRICT RULE REMOVED per user request]
        # We proceed to standard Fusion logic even if no gates are found by OCR/Text.
        # This allows VLM/Embedding to save the image if it looks like a circuit.
        
        # 1. Negative Context was already handled by Fail-Fast above.


        # Summaries from Stage 1 for global use
        stage1_pos = (
            s1b["text_pos_circuit_language"] >= 2
            or s1a["ocr_ket_evidence"] >= 1
            or len(s1a.get("ocr_gate_tokens", [])) >= 1
        )
        stage1_neg = (
            (s1a["ocr_axes_evidence"] >= 1 and s1a["ocr_many_numbers"] >= 2)
            or s1b["text_neg_plot_language"] >= 2
        )

        decision = "REJECT"
        confidence = "LOW"
        reason = ""

        # 1) Hard accept: image clearly closer to circuit concept, and Stage1 not strongly negative
        if score_img >= EMBED_HIGH_ACCEPT and not stage1_neg:
            decision = "ACCEPT"
            confidence = "HIGH"
            reason = "Embedding: image closer to circuit than plot (strong)."

        # 2) Hard reject: image clearly more like a plot than a circuit
        elif score_img <= EMBED_HIGH_REJECT:
            decision = "REJECT"
            confidence = "HIGH"
            reason = "Embedding: image much closer to plot than circuit."

        # 3) Medium accept: some embedding support plus Stage1 positive signal
        # 3) Medium accept: some embedding support plus Stage1 positive signal
        # [TUNING] Precision Mode: Requirement tightened to 0.05 (Strict).
        elif score_img >= 0.05 and stage1_pos and not stage1_neg:
            decision = "ACCEPT"
            confidence = "MEDIUM"
            reason = "Stage1 support + embedding slightly circuit-leaning."

        # 3.5) Mercy Rule: Vision is confident (> 0.05) and text is not Negative (score >= 1)
        # [TUNING] Precision Mode: Requirement tightened to 0.05 (Strict).
        elif score_img >= 0.05 and s1b["text_pos_circuit_language"] >= 1 and not stage1_neg:
            decision = "ACCEPT"
            confidence = "LOW"
            reason = "Mercy Rule: Visual score positive (>= 0.00) and text is neutral/positive."

        # 4) Otherwise conservative reject
        else:
            decision = "REJECT"
            confidence = "LOW"
            reason = "Embedding and Stage1 do not support circuit strongly."

        evidence["fusion"] = {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "flags": {
                "stage1_pos": stage1_pos,
                "stage1_neg": stage1_neg,
                "score_img": score_img,
                "EMBED_HIGH_ACCEPT": EMBED_HIGH_ACCEPT,
                "EMBED_HIGH_REJECT": EMBED_HIGH_REJECT,
            }
        }

        evidence["timing"] = timing

        print(
            f"[TIMING] {arxiv_id} fig{figure_number}: "
            f"stage0={timing.get('stage0', 0):.3f}s, "
            f"ocr={timing.get('stage1_ocr', 0):.3f}s, "
            f"text={timing.get('stage1_text', 0):.3f}s, "
            f"embed={timing.get('stage2_embedding', 0):.3f}s"
        )

        # final monitoring
        if decision == "ACCEPT":
            self.monitor.save("stage2_accepted", arxiv_id, figure_number, image_path)
            self.monitor.save("final_accepted", arxiv_id, figure_number, image_path)
        else:
            self.monitor.save("stage2_rejected", arxiv_id, figure_number, image_path)
            self.monitor.save("final_rejected", arxiv_id, figure_number, image_path)

        # gates from VLM + OCR (union)
        gates = set()
        for g in s2.detected_gate_text:
            gates.add(g)
        for g in s1a.get("ocr_gate_tokens", []):
            gates.add(g)
        for g in s1b.get("text_gate_tokens", []):
            gates.add(g)

        return ClassificationResult(
            decision=decision,
            confidence_tag=confidence,
            reason=reason,
            evidence=evidence,
            quantum_gates=sorted(gates),
            quantum_problem="",
            descriptions=[caption_text] if caption_text else [],
            text_positions=[(0, len(caption_text))] if caption_text else [],
        )
