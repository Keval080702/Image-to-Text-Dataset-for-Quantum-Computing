"""
Integrated Quantum Circuit Dataset Builder
===========================================

End-to-end pipeline:
1. Download arXiv PDFs
2. Extract figures from PDF (using Docling)
3. Classify each figure using 3-stage classifier:
   - Stage 0: Geometric Filter (OpenCV heuristic)
   - Stage 1a: Text Analysis (Caption + Context)
   - Stage 1b: OCR Analysis (Image Text)
   - Stage 2: Vision Judge (CLIP Embeddings)
4. If quantum circuit → save to dataset with metadata
5. Stop when TARGET_CIRCUITS quantum circuits found
6. Generate final outputs (images + dataset JSON + all_classifications JSON)

Author: Exam ID 8
Date: 2025-12-16
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests
from monitoring import StageMonitor
    
# IMPORTANT: allow importing local modules (main_classifier.py) reliably
sys.path.append(str(Path(__file__).resolve().parent))

# Ensure stage monitoring writes to a unified, absolute directory.
# This keeps outputs consistent across scripts and runs.
os.environ["STAGE_MONITOR_DIR"] = str(Path(__file__).resolve().parent / "stage_monitoring")
print(f"[INFO] STAGE_MONITOR_DIR = {os.environ['STAGE_MONITOR_DIR']}")

# Docling for PDF processing
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem

# Optional: Force Docling to NOT use GPU (if you keep getting CUDA OOM in docling layout model)
# Uncomment if needed:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

USE_ADVANCED_CLASSIFIER = True
if USE_ADVANCED_CLASSIFIER:
    try:
        # Use the COPY classifier (embedding-based Stage 2) for experiments
        from main_classifier import QASMClassifier as AdvancedClassifier
        print("[INFO] Using Advanced Classifier: CNN + OCR/Text + Embedding Stage 2")
    except Exception as e:
        print(f"[WARNING] Advanced classifier not available: {e}")
        print("[WARNING] Falling back to simple classifier")
        USE_ADVANCED_CLASSIFIER = False



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
    BASE_DIR = Path(__file__).resolve().parent  # <--- stable base
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
    USE_ADVANCED_CLASSIFIER = USE_ADVANCED_CLASSIFIER


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


class SimpleQuantumCircuitClassifier:
    """
    Fallback simple classifier based strictly on caption keywords.

    This classifier is used if the advanced classifier fails to initialize.
    It performs a simple substring search for standard quantum terms.
    """

    CIRCUIT_KEYWORDS = ["quantum circuit", "circuit diagram", "qubit", "gate sequence", "ansatz", "CNOT", "Hadamard"]

    def classify(self, caption: str, arxiv_id: str, figure_number: int):
        """
        Classify a figure based on its caption.

        Parameters
        ----------
        caption : str
            The text caption of the figure.
        arxiv_id : str
            The arXiv ID of the paper (unused in logic, kept for interface compatibility).
        figure_number : int
            The figure identifier (unused in logic).

        Returns
        -------
        Tuple[bool, List[str], str, str]
            (is_circuit, detected_gates, classification_reason, confidence_level)
        """
        cap = (caption or "").lower()
        if any(k in cap for k in self.CIRCUIT_KEYWORDS):
            return True, [], "Keyword Match", "LOW"
        return False, [], "No Keywords", "LOW_REJECT"


class QuantumCircuitDatasetBuilder:

    def __init__(self, config: Config | None = None):
        """
        Initialize the QuantumCircuitDatasetBuilder.

        Parameters
        ----------
        config : Config, optional
            Configuration object. If None, a default Config is instantiated.
        """
        self.config = config or Config()

        # classifier
        if self.config.USE_ADVANCED_CLASSIFIER:
            try:
                self.classifier = AdvancedClassifier()
                self.use_advanced = True
            except Exception as e:
                print(f"[ERROR] Failed to init AdvancedClassifier: {e}")
                self.classifier = SimpleQuantumCircuitClassifier()
                self.use_advanced = False
        else:
            self.classifier = SimpleQuantumCircuitClassifier()
            self.use_advanced = False

        # docling
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        # Allow overriding via environment variable for stability
        try:
            pipeline_options.images_scale = float(os.environ.get("DOC_IMAGES_SCALE", self.config.IMAGES_SCALE))
        except Exception:
            pipeline_options.images_scale = float(self.config.IMAGES_SCALE)
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        # storage
        self.quantum_circuits: Dict[str, QuantumCircuitEntry] = {}
        self.paper_results: List[PaperProcessingResult] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.all_classifications: List[Dict[str, Any]] = []
        self.paper_list: List[str] = []
        self.paper_counts: Dict[str, int] = {}

        # dirs
        self._create_directories()

    def _create_directories(self):
        """
        Create all necessary output directories if they do not exist.
        """
        self.config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.config.DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self):
        """
        Load and RECONCILE existing progress to ensure consistency.

        This method acts as the "source of truth" synchronizer. It:
        1. Loads the existing dataset JSON.
        2. Scans the 'images' directory.
        3. Removes records from JSON if the image is missing (Prune).
        4. Removes images from Disk if they are not in the JSON (Strict Sync).
        5. Updates the paper processing counts based on valid circuits.
        """
        print("[RESUME] Reconciling State (Images <-> JSON <-> CSV)...")
        
        # 1. Load Dictionary from JSON (if exists)
        loaded_data = {}
        if self.config.DATASET_JSON.exists():
            try:
                with open(self.config.DATASET_JSON, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load dataset JSON: {e}")
        
        # 2. Validate against Disk (Truth)
        valid_circuits = {}
        pruned_count = 0
        
        for fname, d in loaded_data.items():
            img_path = self.config.DATASET_IMAGES_DIR / fname
            if img_path.exists():
                # Reconstruct entry
                entry = QuantumCircuitEntry(
                    image_filename=fname,
                    arxiv_id=d.get("arxiv_id", ""),
                    page_number=d.get("page_number", 0),
                    figure_number=d.get("figure_number", 0),
                    quantum_gates=d.get("quantum_gates", []),
                    quantum_problem=d.get("quantum_problem", ""),
                    descriptions=d.get("descriptions", []),
                    text_positions=d.get("text_positions", [])
                )
                valid_circuits[fname] = entry
            else:
                # Missing file -> Prune
                pruned_count += 1
        
        self.quantum_circuits = valid_circuits
        print(f"[RESUME] Valid Circuits on Disk: {len(self.quantum_circuits)}")
        if pruned_count > 0:
            print(f"[RESUME] Pruned {pruned_count} orphan entries (files not found).")
            # Sync JSON immediately
            self._write_dataset_json()
            print("[RESUME] dataset_8.json Synced.")

        # 2b. Strict Sync: Delete PHYSICAL files not in JSON (User Request: "No Mismatch")
        # This guarantees Folder Count == JSON Count
        if self.config.DATASET_IMAGES_DIR.exists():
            ondisk_files = list(self.config.DATASET_IMAGES_DIR.iterdir())
            removed_orphans = 0
            for p in ondisk_files:
                if p.is_file() and p.name not in self.quantum_circuits:
                    try:
                        os.remove(p)
                        removed_orphans += 1
                        print(f"[RESUME] Deleted orphan image: {p.name}")
                    except OSError as e:
                        print(f"[RESUME] Failed to delete orphan {p.name}: {e}")
            if removed_orphans > 0:
                print(f"[RESUME] Sync complete: Removed {removed_orphans} orphaned files from disk.")

        # 3. Recalculate Paper Counts (Source of Truth for "Done")
        # We assume if a paper is in the CSV, it was "processed".
        # But we must update the specific COUNT to match reality.
        
        # Load existing CSV to know WHICH papers were processed
        processed_papers_set = set()
        if self.config.PAPER_COUNTS_CSV.exists():
            try:
                with open(self.config.PAPER_COUNTS_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        aid = row.get("arxiv_id")
                        c_str = row.get("quantum_circuits_found", "")
                        # [FIX] Only mark as processed if we actually recorded a count (even 0).
                        # Empty string "" means we haven't processed it yet.
                        if aid and c_str.strip(): 
                            processed_papers_set.add(aid)
            except Exception:
                pass
        
        # Recalculate counts strictly from valid_circuits
        current_counts = {}
        for entry in self.quantum_circuits.values():
            aid = entry.arxiv_id
            current_counts[aid] = current_counts.get(aid, 0) + 1
            
        # Update self.paper_counts map
        # If a paper was processed but now has 0 circuits (all rejected), it keeps 0.
        # If a paper was never processed, it stays None.
        for arxiv_id in processed_papers_set:
            self.paper_counts[arxiv_id] = current_counts.get(arxiv_id, 0)
            
        # 4. Rewrite consistency CSV
        # This ensures the CSV doesn't report "10 found" when we really have 2.
        if processed_papers_set:
            self._write_outputs()  # This uses self.paper_counts to write the CSV
            print("[RESUME] paper_list_counts_8.csv Synced.")
            
        processed_count = sum(1 for v in self.paper_counts.values() if v is not None)
        print(f"[RESUME] Pipeline Ready. {processed_count} papers tracked.")


        
        # Original deletion logic DISABLED:
        # try:
        #     metadata_full = self.config.DATASET_DIR / f"metadata_full_{self.config.EXAM_ID}.json"
        #     for p in ...


    def build_dataset(self):
        """
        Execute the main dataset building pipeline.

        This method:
        1. Reads the list of arXiv papers to process.
        2. Loads prior progress from checkpoint (skipping already processed papers).
        3. Iteratively processes each paper:
           - Downloads PDF.
           - Extracts figures.
           - Classifies figures.
           - Saves accepted circuit images.
        4. Stops when TARGET_CIRCUITS is reached.
        5. Writes summary outputs (CSV, JSON).
        """
        print("=" * 80)
        print(f"QUANTUM CIRCUIT DATASET BUILDER - EXAM ID {self.config.EXAM_ID}")
        print(f"Target: {self.config.TARGET_CIRCUITS} quantum circuits")
        print("=" * 80)

        self.paper_list = self._read_paper_list()
        # None = not looked into yet; integer = analyzed paper (may be 0)
        self.paper_counts = {arxiv_id: None for arxiv_id in self.paper_list}
        print(f"[INFO] Found {len(self.paper_list)} papers in list")

        self._load_checkpoint()
        
        # [FIX] Initialize total_circuits from loaded data
        total_circuits = len(self.quantum_circuits)
        print(f"[INFO] Starting circuit count: {total_circuits}/{self.config.TARGET_CIRCUITS}")

        for i, arxiv_id in enumerate(self.paper_list, start=1):
            # [RESUME] Skip if already processed
            if self.paper_counts.get(arxiv_id) is not None:
                continue

            print("\n" + "=" * 80)
            print(f"Processing paper {i}/{len(self.paper_list)}: {arxiv_id}")
            print(f"Quantum circuits found so far: {total_circuits}/{self.config.TARGET_CIRCUITS}")
            print("=" * 80)

            result = self.process_paper(arxiv_id)
            self.paper_results.append(result)
            # Mark this paper as analyzed with the exact number found
            self.paper_counts[arxiv_id] = result.quantum_circuits_found
            total_circuits += result.quantum_circuits_found

            print(f"\n[Paper {i}] Status: {result.status}")
            print(f"[Paper {i}] Figures extracted: {result.total_figures_extracted}")
            print(f"[Paper {i}] Quantum circuits: {result.quantum_circuits_found}")

            if total_circuits >= self.config.TARGET_CIRCUITS:
                print("\n" + "=" * 80)
                print(f"TARGET REACHED: {total_circuits}")
                print("=" * 80)
                break

            if i < len(self.paper_list) and total_circuits < self.config.TARGET_CIRCUITS:
                print(f"\nWaiting {self.config.SLEEP_BETWEEN_PAPERS} seconds before next paper...\n")
                # [CRITICAL] Save Progress Incrementally (CSV + JSON)
                # This ensures we don't lose the "Checked Papers" list if stopped specificially.
                self._write_outputs()
                time.sleep(self.config.SLEEP_BETWEEN_PAPERS)

        self._write_outputs()
        self._print_summary()

    def process_paper(self, arxiv_id: str) -> PaperProcessingResult:
        """
        Process a single arXiv paper to find quantum circuits.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier (e.g., "2305.12345").

        Returns
        -------
        PaperProcessingResult
            Summary of the processing outcome, including counts of figures found/accepted.

        Notes
        -----
        This method handles the full lifecycle for one paper:
        - PDF Download
        - Docling conversion/extraction
        - Classification (Stage 0 -> Stage 1 -> Stage 2)
        - Result accumulation
        """
        pdf_path = self._download_pdf(arxiv_id)
        if not pdf_path:
            return PaperProcessingResult(arxiv_id, None, "download_failed", 0, 0, "Failed to download PDF")

        try:
            print(f"[{arxiv_id}] Converting PDF with Docling...")
            conv_res = self.doc_converter.convert(str(pdf_path))
            doc = conv_res.document

            figures = self._extract_figures(doc, arxiv_id, str(pdf_path))
            print(f"[{arxiv_id}] Extracted {len(figures)} figures")

            quantum_found = 0
            for fig in figures:
                # Stop immediately if we already have enough saved circuits globally
                if len(self.quantum_circuits) >= self.config.TARGET_CIRCUITS:
                    print(
                        f"  [INFO] Global target {self.config.TARGET_CIRCUITS} reached; "
                        "skipping remaining figures in this paper."
                    )
                    break

                if self.use_advanced:
                    try:
                        res = self.classifier.classify(
                            arxiv_id=arxiv_id,
                            page_number=fig["page_number"],
                            figure_number=fig["figure_number"],
                            image_path=fig["image_path"],
                            caption_text=fig["caption"],
                            pdf_path=str(pdf_path),
                        )
                        is_circuit = (res.decision == "ACCEPT")
                        gates = res.quantum_gates
                        problem = res.quantum_problem
                        confidence = res.confidence_tag
                    except Exception as e:
                        print(f"  [ERROR] Advanced classifier failed for Fig {fig['figure_number']}: {e}")
                        is_circuit, gates, problem, confidence = False, [], "", "ERROR"
                else:
                    is_circuit, gates, problem, confidence = self.classifier.classify(
                        fig["caption"], arxiv_id, fig["figure_number"]
                    )

                # Fallback: if monitor recorded final acceptance, store image+JSON regardless
                if not is_circuit:
                    try:
                        if self._monitor_final_accepted(arxiv_id, fig["figure_number"]):
                            is_circuit = True
                            # keep gates/problem from classifier (may be empty) — metadata handles gate rejection
                            confidence = "MONITOR_ACCEPT"
                            print("  [INFO] Overriding to ACCEPT due to monitor final_accepted.")
                    except Exception:
                        pass

                self.all_classifications.append({
                    "arxiv_id": arxiv_id,
                    "page_number": fig["page_number"],
                    "figure_number": fig["figure_number"],
                    "image_path": fig["image_path"],
                    "caption": fig["caption"],
                    "decision": "ACCEPT" if is_circuit else "REJECT",
                    "confidence": confidence,
                    "quantum_gates": gates if is_circuit else [],
                    "quantum_problem": problem if is_circuit else "",
                    "classifier_type": "advanced" if self.use_advanced else "simple"
                })

                self.audit_log.append({
                    "arxiv_id": arxiv_id,
                    "figure_number": fig["figure_number"],
                    "is_quantum_circuit": is_circuit,
                    "confidence": confidence,
                    "gates": gates,
                    "problem": problem,
                    "caption": (fig["caption"] or "")[:200],
                    "classifier_type": "advanced" if self.use_advanced else "simple"
                })

                if is_circuit:
                    self._add_to_dataset(fig, gates, problem)
                    quantum_found += 1
                    print(f"  ✓ Figure {fig['figure_number']}: QUANTUM CIRCUIT (confidence: {confidence})")
                else:
                    print(f"  ✗ Figure {fig['figure_number']}: Not a circuit")

            # After processing all figures from this paper, ingest any monitor-driven final acceptances
            try:
                self._ingest_monitor_acceptances(arxiv_id, figures)
            except Exception as e:
                print(f"  [WARN] Monitor ingestion failed: {e}")

            return PaperProcessingResult(arxiv_id, str(pdf_path), "ok", len(figures), quantum_found)

        except Exception as e:
            print(f"[{arxiv_id}] ERROR: {e}")
            return PaperProcessingResult(arxiv_id, str(pdf_path), "processing_error", 0, 0, str(e))

    def _download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """
        Download the PDF for a given arXiv ID.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier.

        Returns
        -------
        Optional[Path]
            Path to the downloaded PDF file, or None if download failed.
        """
        safe_id = arxiv_id.replace("/", "_")
        out_path = self.config.PDF_DIR / f"{safe_id}.pdf"
        if out_path.exists():
            print(f"[{arxiv_id}] PDF already exists")
            return out_path

        url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[{arxiv_id}] Downloading from {url}")

        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                out_path.write_bytes(resp.content)
                print(f"[{arxiv_id}] Downloaded successfully")
                return out_path
        except Exception as e:
            print(f"[{arxiv_id}] Download error: {e}")
        return None

    def _clean_description(self, text: str) -> str:
        """Remove leading "Fig.", "FIG.", "Figure" prefixes from a description.

        Examples:
            - "FIG. 3: ..." -> "..."
            - "Fig. 2. ..." -> "..."
            - "Figure 5 - ..." -> "..."
        """
        if not text:
            return ""

        s = text.strip()
        # Remove common figure prefixes with numbers, including supplementary like S1
        pattern_num = r"^(?:fig(?:ure)?\.?|figure)\s*(?:s)?\s*\d+[a-zA-Z0-9]*[:\.\-]*\s*"
        s = re.sub(pattern_num, "", s, flags=re.IGNORECASE)
        # Also remove bare prefixes like "Fig." or "FIG:" if present at start
        pattern_bare = r"^(?:fig(?:ure)?\.?|figure)[:\.\-]*\s*"
        s = re.sub(pattern_bare, "", s, flags=re.IGNORECASE)

        return s.strip()

    def _extract_figures(self, doc: Any, arxiv_id: str, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract figures from a Docling-converted document.

        Parameters
        ----------
        doc : Any
            The docling document object.
        arxiv_id : str
            The source arXiv ID.
        pdf_path : str
            Path to the source PDF.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, one per extracted figure, containing:
            - page_number
            - figure_number
            - caption
            - image_path
            - context_mentions (full page text)
        """
        figures: List[Dict[str, Any]] = []

        for pic_idx, pic in enumerate(getattr(doc, "pictures", []), start=1):
            if not isinstance(pic, PictureItem):
                continue
            prov_list = getattr(pic, "prov", None)
            if not prov_list:
                continue
            page_number = int(getattr(prov_list[0], "page_no", 1))

            # Extract caption AND full page text context for better problem ID
            caption, page_full_text = self._get_caption_and_context(pic, doc, pdf_path)

            figure_number = self._extract_figure_number(caption)
            if figure_number is None:
                figure_number = -pic_idx

            image_path = self._save_figure_image(pic, doc, arxiv_id, page_number, pic_idx)
            if image_path is None:
                continue

            figures.append({
                "arxiv_id": arxiv_id,
                "page_number": page_number,
                "figure_number": figure_number,
                "caption": caption,
                "image_path": image_path,
                "pic_index": pic_idx,
                "pdf_path": pdf_path,
                "context_mentions": [page_full_text], # Pass full text here
            })

        return figures

    def _get_caption_and_context(self, pic: PictureItem, doc: Any, pdf_path: str) -> Tuple[str, str]:
        """
        Retrieve the caption and surrounding text context for a figure.

        This method attempts to get the caption from Docling metadata.
        If missing, it performs a spatial search using PyMuPDF to find text
        immediately above or below the figure's bounding box.

        Parameters
        ----------
        pic : PictureItem
            The figure item from Docling.
        doc : Any
            The docling document.
        pdf_path : str
            Path to the source PDF.

        Returns
        -------
        Tuple[str, str]
            (Caption text, Full page text)
        """
        cap_attr = getattr(pic, "caption_text", None)
        caption = ""

        if callable(cap_attr):
            try:
                caption = cap_attr(doc) or ""
            except Exception:
                try:
                    caption = cap_attr() or ""
                except Exception:
                    caption = ""
        elif isinstance(cap_attr, str):
            caption = cap_attr
        
        # We need to open PDF to get full text anyway
        prov_list = getattr(pic, "prov", None)
        if not prov_list:
            return caption if caption else "Figure", ""

        page_index = int(getattr(prov_list[0], "page_no", 1)) - 1
        
        try:
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(pdf_path)
            if page_index < 0 or page_index >= len(pdf_doc):
                pdf_doc.close()
                return caption if caption else "Figure", ""

            page = pdf_doc[page_index]
            text = page.get_text("text") or ""
            
            # Use blocks for spatial search
            blocks = page.get_text("blocks")
            
            pdf_doc.close()

            if caption and len(caption.strip()) > 10:
                 return caption, text
            
            print("  [Caption] No caption found, searching nearby text spatially...")
            
            # Spatial Search: Find text below, then above
            # docling prov gives: [ProvenanceItem(page_no=1, bbox=[x0, y0, x1, y1])]
            # PyMuPDF bbox: [x0, y0, x1, y1] (top-left, bottom-right)
            
            fig_bbox = None
            if prov_list and hasattr(prov_list[0], 'bbox'):
                b = prov_list[0].bbox
                # docling coords: x0, y0, x1, y1 usually
                fig_bbox = b
                
            # Filter for text blocks (type 0)
            text_blocks = [b for b in blocks if b[6] == 0 and len(b[4].strip()) > 20]
            
            if not text_blocks or not fig_bbox:
                # Fallback to simple logic if no bbox info
                 paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
                 fallback = paragraphs[0] if paragraphs else "Figure"
                 if paragraphs:
                     print(f"  [Caption] Using simple fallback: {fallback[:50]}...")
                 return fallback, text
            
            # Docling bbox: [l, b, r, t] or [l, t, r, b]? Assuming [x0, y0, x1, y1] (top-left, bottom-right)
            # PyMuPDF get_text("blocks") uses standard PDF coords (y=0 at top)
            # Docling v2 uses bottom-left origin? NO, Docling v2 is usually top-left.
            # We will try robustly.
            
            fig_y_bottom = fig_bbox[3]
            fig_y_top = fig_bbox[1]
            
            # Find candidates below
            below_candidates = []
            above_candidates = []
            
            for b_idx, b in enumerate(text_blocks):
                bx0, by0, bx1, by1, btext, _, _ = b
                # b[4] is text
                
                # Check 1: Is this block definitely below the figure?
                # i.e. block_top > fig_bottom
                if by0 > fig_y_bottom: 
                    below_candidates.append((b, by0 - fig_y_bottom))
                
                # Check 2: Is this block definitely above the figure?
                # i.e. block_bottom < fig_top
                elif by1 < fig_y_top:
                    above_candidates.append((b, fig_y_top - by1))
                    
            # Sort by proximity
            below_candidates.sort(key=lambda x: x[1])
            above_candidates.sort(key=lambda x: x[1])
            
            best_text = ""
            
            # Priority 1: Below (User: "look below text")
            if below_candidates:
                # Try to find one that starts with "Fig"
                for cand in below_candidates[:3]:
                    txt = cand[0][4].strip()
                    if re.match(r"(?:Figure|Fig\.?|FIG\.?)\s*\d+", txt, re.IGNORECASE):
                         best_text = txt
                         print(f"  [Caption] Found caption BELOW: {best_text[:50]}...")
                         break
                if not best_text:
                    # Just take the closest text block
                    best_text = below_candidates[0][0][4].strip()
                    print(f"  [Caption] Using nearest text BELOW: {best_text[:50]}...")
                    
            # Priority 2: Above (User: "than look fore above text")
            elif above_candidates:
                for cand in above_candidates[:3]:
                    txt = cand[0][4].strip()
                    if re.match(r"(?:Figure|Fig\.?|FIG\.?)\s*\d+", txt, re.IGNORECASE):
                         best_text = txt
                         print(f"  [Caption] Found caption ABOVE: {best_text[:50]}...")
                         break
                if not best_text:
                    best_text = above_candidates[0][0][4].strip()
                    print(f"  [Caption] Using nearest text ABOVE: {best_text[:50]}...")
            
            if best_text:
                return best_text, text
                
            # If nothing strictly above/below found?
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            fallback = paragraphs[0] if paragraphs else "Figure"
            print(f"  [Caption] Spatial search failed, using simple fallback: {fallback[:50]}...")
            return fallback, text

        except Exception as e:
            print(f"  [Caption] Fallback extraction failed: {e}")

        return caption if caption else "Figure", ""

    def _extract_figure_number(self, caption: str) -> Optional[int]:
        """
        Parse the figure number from the caption text (e.g., "Figure 3").
        """
        m = re.search(r"(?:Figure|Fig\.?|FIG\.?)\s*(\d+)", caption or "", re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _save_figure_image(self, pic: PictureItem, doc: Any, arxiv_id: str, page_number: int, pic_idx: int) -> Optional[str]:
        """
        Rasterize and save the figure image to disk.

        Parameters
        ----------
        pic : PictureItem
            The figure item.
        doc : Any
            The document object.
        arxiv_id : str
            The arXiv ID.
        page_number : int
            The page number.
        pic_idx : int
            The internal picture index.

        Returns
        -------
        Optional[str]
            Absolute path to the saved image, or None if save failed/skipped.
        """
        safe_id = arxiv_id.replace("/", "_")
        out_dir = self.config.FIGURES_DIR / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{safe_id}_p{page_number:03d}_f{pic_idx:03d}.png"
        image_path = out_dir / filename

        try:
            img = pic.get_image(doc)
            if img is None:
                return None

            # Filter out tiny thumbnails/icons before saving
            min_dim = min(img.size)
            if min_dim < 64:
                print(f"  [Figure] Skipping tiny image ({img.size[0]}x{img.size[1]} px)")
                return None

            img.save(str(image_path), format="PNG")
            return str(image_path)
        except Exception:
            return None

    def _add_to_dataset(self, fig: Dict[str, Any], gates: List[str], problem: str):
        """
        Add a confirmed quantum circuit to the dataset.

        Parameters
        ----------
        fig : Dict[str, Any]
            The figure metadata dictionary.
        gates : List[str]
            List of detected quantum gates.
        problem : str
            Identified problem domain.

        Notes
        -----
        This method copies the image to the final dataset folder,
        computes global text positions, and adds the entry to the JSON registry.
        """
        safe_id = fig["arxiv_id"].replace("/", "_")
        # Use the original extracted image filename (page + picture index)
        # to guarantee uniqueness and avoid overwriting entries.
        from pathlib import Path as _Path

        image_filename = _Path(fig["image_path"]).name
        dst_path = self.config.DATASET_IMAGES_DIR / image_filename

        copied_ok = False
        try:
            import shutil
            shutil.copy(fig["image_path"], str(dst_path))
            copied_ok = dst_path.exists()
        except Exception as e:
            print(f"  [WARN] Failed copying dataset image: {e}")
            copied_ok = False

        # Fallback: copy from stage_monitoring/final_accepted if available
        if not copied_ok:
            try:
                safe_id = fig["arxiv_id"].replace("/", "_")
                monitor_root = Path(os.environ.get("STAGE_MONITOR_DIR", str(self.config.BASE_DIR / "stage_monitoring")))
                fa_dir = monitor_root / "final_accepted"
                if fa_dir.exists():
                    # Try common extensions
                    for ext in [".png", ".jpg", ".jpeg"]:
                        src_candidate = fa_dir / f"{safe_id}_fig{fig['figure_number']}{ext}"
                        if src_candidate.exists():
                            shutil.copy(str(src_candidate), str(dst_path))
                            copied_ok = dst_path.exists()
                            if copied_ok:
                                print(f"  [INFO] Fallback copied from monitor: {src_candidate.name} -> {dst_path.name}")
                                break
            except Exception as e:
                print(f"  [WARN] Fallback copy from monitor failed: {e}")

        if not copied_ok:
            # Do not create JSON entry unless image copy succeeded
            print("  [WARN] Skipping dataset JSON entry because image copy failed.")
            return

        caption = fig.get("caption", "") or ""
        # Clean figure prefixes globally (e.g., "Fig.", "Figure", "FIG.")
        caption_cleaned = self._clean_description(caption)

        # Normalize quantum problem: if empty, store explicit "unknown"
        qp = (problem or "").strip()
        if not qp:
            qp = "unknown"
        # Compute GLOBAL token-based text positions for the caption across the entire PDF
        positions: List[Tuple[int, int]] = []
        try:
            pdf_path = fig.get("pdf_path") or ""
            if pdf_path and caption_cleaned.strip():
                positions = self._compute_global_text_positions(pdf_path, [caption_cleaned])
        except Exception as e:
            print(f"  [WARN] Failed computing global text positions: {e}")
        entry = QuantumCircuitEntry(
            image_filename=image_filename,
            arxiv_id=fig["arxiv_id"],
            page_number=fig["page_number"],
            figure_number=fig["figure_number"],
            quantum_gates=gates or [],
            quantum_problem=qp,
            descriptions=[caption_cleaned] if caption_cleaned else [],
            text_positions=positions or [],
        )
        self.quantum_circuits[image_filename] = entry

        # Immediately update the dataset JSON alongside image selection
        try:
            self._write_dataset_json()
            # Verify JSON now contains this entry
            try:
                with open(self.config.DATASET_JSON, "r", encoding="utf-8") as _jf:
                    _data = json.load(_jf)
                if image_filename in _data:
                    print(f"  [DATASET] JSON updated with {image_filename}")
                else:
                    print(f"  [WARN] JSON does not contain {image_filename} (unexpected)")
            except Exception as _ve:
                print(f"  [WARN] Could not verify dataset JSON entry: {_ve}")
            # Verify image exists
            if dst_path.exists():
                print(f"  [DATASET] Image stored: {dst_path}")
            else:
                print(f"  [WARN] Image not found after copy: {dst_path}")
        except Exception as e:
            print(f"  [WARN] Failed writing incremental dataset JSON: {e}")

    def _write_dataset_json(self):
        """Write the current dataset JSON snapshot.

        Generates the same structure as at the end, so consumers can
        read the dataset during processing. Called incrementally
        whenever a figure is accepted and at finalization.
        """
        self.config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        dataset_dict: Dict[str, Any] = {}
        for filename, entry in self.quantum_circuits.items():
            # [STRICT SYNC] Integrity Check: Do not write entry if file is missing
            if not (self.config.DATASET_IMAGES_DIR / filename).exists():
                continue

            d = asdict(entry)
            d.pop("image_filename", None)
            dataset_dict[filename] = d

        sorted_keys = sorted(dataset_dict.keys())
        sorted_dataset = {k: dataset_dict[k] for k in sorted_keys}

        with open(self.config.DATASET_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted_dataset, f, indent=2, ensure_ascii=False)
            # Ensure the JSON is persisted immediately (atomic feel)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass

    def _monitor_final_accepted(self, arxiv_id: str, figure_number: int) -> bool:
        """
        Check if the stage monitor has recorded a final acceptance for this figure.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier.
        figure_number : int
            The figure number.

        Returns
        -------
        bool
            True if the figure file exists in the 'final_accepted' monitor folder.
        """
        safe_id = arxiv_id.replace("/", "_")
        root = Path(os.environ.get("STAGE_MONITOR_DIR", str(self.config.BASE_DIR / "stage_monitoring")))
        dir_path = root / "final_accepted"
        if not dir_path.exists():
            return False
        patterns = [
            f"{safe_id}_fig{figure_number}.png",
            f"{safe_id}_fig{figure_number}.jpg",
            f"{safe_id}_fig{figure_number}.jpeg",
        ]
        for name in patterns:
            if (dir_path / name).exists():
                return True
        return False

    def _ingest_monitor_acceptances(self, arxiv_id: str, figures: List[Dict[str, Any]]) -> None:
        """
        Ingest figures flagged as 'final_accepted' by external monitoring.

        This ensures images manually moved or flagged in the monitor folder are
        added to the dataset even if the pipeline logic initially rejected them.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier.
        figures : List[Dict[str, Any]]
            List of extracted figure dictionaries for this paper.
        """
        safe_id = arxiv_id.replace("/", "_")
        root = Path(os.environ.get("STAGE_MONITOR_DIR", str(self.config.BASE_DIR / "stage_monitoring")))
        dir_path = root / "final_accepted"
        if not dir_path.exists():
            return

        # Build map figure_number -> fig dict
        fig_map: Dict[int, Dict[str, Any]] = {}
        for f in figures:
            try:
                fn = int(f.get("figure_number", -1))
                if fn != -1:
                    fig_map[fn] = f
            except Exception:
                continue

        # Scan final_accepted files for this arxiv_id
        for p in dir_path.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if not name.startswith(f"{safe_id}_fig"):
                continue
            m = re.match(rf"{re.escape(safe_id)}_fig(\d+)\.(png|jpg|jpeg)$", name, re.IGNORECASE)
            if not m:
                continue
            fig_num = int(m.group(1))
            fig = fig_map.get(fig_num)
            if not fig:
                # No matching extracted figure; skip (cannot build JSON without metadata)
                continue

            image_filename = Path(fig["image_path"]).name
            if image_filename in self.quantum_circuits:
                # Already stored
                continue

            print(f"  [INFO] Ingesting monitor acceptance for Fig {fig_num} -> {image_filename}")
            # Store with empty gates/problem; metadata builder will enrich or reject later
            self._add_to_dataset(fig, gates=[], problem="")

    def _compute_global_text_positions(self, pdf_path: str, descriptions: List[str]) -> List[Tuple[int, int]]:
        """
        Compute GLOBAL token positions for descriptions across the entire PDF.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        descriptions : List[str]
            List of text descriptions (captions) to locate.

        Returns
        -------
        List[Tuple[int, int]]
            List of (start_token_index, end_token_index) tuples.
        """
        import pymupdf
        import re as _re

        positions: List[Tuple[int, int]] = []
        if not pdf_path or not descriptions:
            return positions

        try:
            doc = pymupdf.open(pdf_path)

            # Build global token list
            global_tokens: List[str] = []
            for p_idx in range(len(doc)):
                page_text = doc[p_idx].get_text()
                global_tokens.extend(page_text.split())
            doc.close()

            # Normalization similar to metadata generator
            def _norm(t: str) -> str:
                t = t.lower()
                t = _re.sub(r"[^\w\s]", "", t)
                return t

            global_norm = [_norm(t) for t in global_tokens]
            n_global = len(global_norm)

            for desc in descriptions:
                if not desc:
                    continue
                desc_tokens = desc.split()
                if not desc_tokens:
                    continue
                desc_norm = [_norm(t) for t in desc_tokens]
                n_desc = len(desc_norm)

                # Anchor search: try 8,6,4,3-token anchors from the start
                found_start = -1
                for try_len in [8, 6, 4, 3]:
                    if try_len > n_desc:
                        continue
                    anchor = desc_norm[:try_len]
                    start_token = anchor[0]
                    candidates = [i for i, t in enumerate(global_norm) if t == start_token]
                    for i in candidates:
                        if i + try_len > n_global:
                            continue
                        match = True
                        for k in range(1, try_len):
                            if global_norm[i + k] != anchor[k]:
                                match = False
                                break
                        if match:
                            found_start = i
                            break
                    if found_start != -1:
                        break

                if found_start != -1:
                    end_idx = min(found_start + n_desc - 1, n_global - 1)
                    positions.append((found_start, end_idx))
        except Exception as e:
            print(f"  [WARN] Global token match error: {e}")

        return positions


    def _read_paper_list(self) -> List[str]:
        """
        Read the list of arXiv IDs from the configured text file.

        Returns
        -------
        List[str]
            List of arXiv IDs (e.g., "2305.12345").
        """
        papers: List[str] = []
        with open(self.config.PAPER_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("arXiv:"):
                    s = s[6:]
                papers.append(s)
        return papers

    def _write_outputs(self):
        """
        Write all dataset outputs (JSON, CSV, Metadata) to disk.

        This method is called periodically (incremental save) and at the end.
        It updates:
        - dataset_8.json (Main Dataset)
        - paper_list_counts_8.csv (Progress Tracking)
        - metadata_full_8.json (Consolidated Metadata)
        """
        # Ensure the dataset JSON is up to date (same logic as incremental)
        self._write_dataset_json()
        try:
            # Count entries for logging
            with open(self.config.DATASET_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✓ Wrote dataset JSON: {self.config.DATASET_JSON} ({len(data)} circuits)")
        except Exception:
            print(f"✓ Wrote dataset JSON: {self.config.DATASET_JSON}")

        with open(self.config.PAPER_COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arxiv_id", "quantum_circuits_found"])
            # For papers we actually analyzed, write the exact count
            # (including 0). For papers we never looked into (because the
            # target was reached earlier), leave the value blank.
            for arxiv_id in self.paper_list:
                count = self.paper_counts.get(arxiv_id, None)
                if count is None:
                    w.writerow([arxiv_id, ""])
                else:
                    w.writerow([arxiv_id, count])
        print(f"✓ Wrote paper counts CSV: {self.config.PAPER_COUNTS_CSV}")

        # Build the single, final metadata JSON for all images and
        # remove the intermediate JSONs so that only one JSON file is
        # considered the main output.
        try:
            import metadata_generator
            build_metadata_from_dataset = metadata_generator.build_metadata_from_dataset

            print("[INFO] Building consolidated metadata JSON (metadata_full_8.json)...")
            build_metadata_from_dataset()
            # Keep dataset JSON intact; write metadata to its own file.
            metadata_full = self.config.DATASET_DIR / f"metadata_full_{self.config.EXAM_ID}.json"
            if metadata_full.exists():
                print(f"[INFO] Consolidated metadata available at: {metadata_full}")
        except Exception as e:
            print(f"[WARN] Failed to build consolidated metadata JSON: {e}")

    def _print_summary(self):
        """
        Print a final summary of the dataset building process to stdout.
        """
        total_circuits = len(self.quantum_circuits)
        total_figs = len(self.all_classifications)
        print("\n" + "=" * 80)
        print("DATASET BUILDING COMPLETE")
        print("=" * 80)
        print(f"Papers processed: {len(self.paper_results)}")
        print(f"Total figures processed: {total_figs}")
        print(f"Quantum circuits accepted: {total_circuits}")
        print("=" * 80)


def main():
    """
    Main entry point for the Quantum Circuit Dataset Builder.

    Parses command-line arguments, initializes the builder, and executes the pipeline.
    """
    parser = argparse.ArgumentParser(description="Integrated Quantum Circuit Dataset Builder")
    parser.add_argument("--target-circuits", type=int, default=Config.TARGET_CIRCUITS, help="Number of circuits to collect")
    parser.add_argument("--papers-limit", type=int, default=None, help="Limit number of papers processed")
    parser.add_argument("--no-sleep", action="store_true", help="Disable sleep between papers")
    parser.add_argument("--monitor-dir", type=str, default=None, help="Override stage monitoring directory (env STAGE_MONITOR_DIR)")
    parser.add_argument("--monitor-verbose", action="store_true", help="Enable verbose stage monitoring output")
    args = parser.parse_args()

    cfg = Config()
    cfg.TARGET_CIRCUITS = int(args.target_circuits)

    # Apply monitoring env overrides before classifier init
    if args.monitor_dir:
        os.environ["STAGE_MONITOR_DIR"] = args.monitor_dir
        print(f"[INFO] Overriding STAGE_MONITOR_DIR = {os.environ['STAGE_MONITOR_DIR']}")
    if args.monitor_verbose:
        os.environ["STAGE_MONITOR_VERBOSE"] = "1"
        print("[INFO] STAGE_MONITOR_VERBOSE enabled")

    # Initialize monitoring folders (final value after any overrides)
    _monitor_root = Path(os.environ["STAGE_MONITOR_DIR"]).resolve()
    _monitor_root.mkdir(parents=True, exist_ok=True)
    # Create monitoring directories (swapped 1a/1b to match execution order)
    for _name in [
        "stage0_accepted","stage0_rejected",
        "stage1_text_accepted","stage1_text_rejected",
        "stage1_ocr_accepted","stage1_ocr_rejected",
        "stage2_accepted","stage2_rejected",
        "final_accepted","final_rejected",
    ]:
        (_monitor_root / _name).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Monitoring folders initialized at: {_monitor_root}")

    builder = QuantumCircuitDatasetBuilder(cfg)

    # If limiting papers, slice the paper list after reading
    # We need to rebuild the list once; easiest: monkey-patch build flow
    orig_read = builder._read_paper_list
    def _read_limited():
        lst = orig_read()
        if args.papers_limit is not None:
            return lst[: max(0, int(args.papers_limit))]
        return lst
    builder._read_paper_list = _read_limited

    # Optionally disable sleep by overriding config
    if args.no_sleep:
        builder.config.SLEEP_BETWEEN_PAPERS = 0

    builder.build_dataset()


if __name__ == "__main__":
    main()
