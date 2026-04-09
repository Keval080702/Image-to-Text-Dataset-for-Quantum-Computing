"""
Dataset Builder Module
=======================

Main dataset building orchestration and logic.

This module contains the QuantumCircuitDatasetBuilder class which orchestrates
the entire pipeline: PDF downloading, figure extraction, classification, and dataset creation.
"""

from __future__ import annotations
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from config import Config
from models import QuantumCircuitEntry, PaperProcessingResult
from pdf_processor import PDFProcessor
from classifiers import QASMClassifier


class SimpleQuantumCircuitClassifier:
    """
    Fallback simple classifier based strictly on caption keywords.
    
    This classifier is used if the advanced classifier fails to initialize.
    """

    CIRCUIT_KEYWORDS = ["quantum circuit", "circuit diagram", "qubit", "gate sequence", "ansatz", "CNOT", "Hadamard"]

    def classify(self, caption: str, arxiv_id: str, figure_number: int):
        """Classify a figure based on its caption."""
        cap = (caption or "").lower()
        if any(k in cap for k in self.CIRCUIT_KEYWORDS):
            return True, [], "Keyword Match", "LOW"
        return False, [], "No Keywords", "LOW_REJECT"


class QuantumCircuitDatasetBuilder:
    """Main dataset builder class."""

    def __init__(self, config: Config | None = None):
        """Initialize the QuantumCircuitDatasetBuilder."""
        self.config = config or Config()

        # Initialize classifier
        if self.config.USE_ADVANCED_CLASSIFIER:
            try:
                self.classifier = QASMClassifier()
                self.use_advanced = True
            except Exception as e:
                print(f"[ERROR] Failed to init AdvancedClassifier: {e}")
                self.classifier = SimpleQuantumCircuitClassifier()
                self.use_advanced = False
        else:
            self.classifier = SimpleQuantumCircuitClassifier()
            self.use_advanced = False

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(self.config)

        # Storage
        self.quantum_circuits: Dict[str, QuantumCircuitEntry] = {}
        self.paper_results: List[PaperProcessingResult] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.all_classifications: List[Dict[str, Any]] = []
        self.paper_list: List[str] = []
        self.paper_counts: Dict[str, int] = {}

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary output directories."""
        self.config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        self.config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.config.DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self):
        """Load and reconcile existing progress."""
        print("[RESUME] Reconciling State (Images <-> JSON <-> CSV)...")
        
        # Load dictionary from JSON
        loaded_data = {}
        if self.config.DATASET_JSON.exists():
            try:
                with open(self.config.DATASET_JSON, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load dataset JSON: {e}")
        
        # Validate against disk
        valid_circuits = {}
        pruned_count = 0
        
        for fname, d in loaded_data.items():
            img_path = self.config.DATASET_IMAGES_DIR / fname
            if img_path.exists():
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
                pruned_count += 1
        
        self.quantum_circuits = valid_circuits
        print(f"[RESUME] Valid Circuits on Disk: {len(self.quantum_circuits)}")
        if pruned_count > 0:
            print(f"[RESUME] Pruned {pruned_count} orphan entries.")
            self._write_dataset_json()

        # Sync orphaned files
        if self.config.DATASET_IMAGES_DIR.exists():
            ondisk_files = list(self.config.DATASET_IMAGES_DIR.iterdir())
            removed_orphans = 0
            for p in ondisk_files:
                if p.is_file() and p.name not in self.quantum_circuits:
                    try:
                        os.remove(p)
                        removed_orphans += 1
                    except OSError:
                        pass
            if removed_orphans > 0:
                print(f"[RESUME] Removed {removed_orphans} orphaned files from disk.")

        # Load processed papers
        processed_papers_set = set()
        if self.config.PAPER_COUNTS_CSV.exists():
            try:
                with open(self.config.PAPER_COUNTS_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        aid = row.get("arxiv_id")
                        c_str = row.get("quantum_circuits_found", "")
                        if aid and c_str.strip():
                            processed_papers_set.add(aid)
            except Exception:
                pass
        
        # Recalculate counts
        current_counts = {}
        for entry in self.quantum_circuits.values():
            aid = entry.arxiv_id
            current_counts[aid] = current_counts.get(aid, 0) + 1
            
        for arxiv_id in processed_papers_set:
            self.paper_counts[arxiv_id] = current_counts.get(arxiv_id, 0)
            
        if processed_papers_set:
            self._write_outputs()
            
        processed_count = sum(1 for v in self.paper_counts.values() if v is not None)
        print(f"[RESUME] Pipeline Ready. {processed_count} papers tracked.")

    def build_dataset(self):
        """Execute the main dataset building pipeline."""
        print("=" * 80)
        print(f"QUANTUM CIRCUIT DATASET BUILDER - EXAM ID {self.config.EXAM_ID}")
        print(f"Target: {self.config.TARGET_CIRCUITS} quantum circuits")
        print("=" * 80)

        self.paper_list = self._read_paper_list()
        self.paper_counts = {arxiv_id: None for arxiv_id in self.paper_list}
        print(f"[INFO] Found {len(self.paper_list)} papers in list")

        self._load_checkpoint()
        
        total_circuits = len(self.quantum_circuits)
        print(f"[INFO] Starting circuit count: {total_circuits}/{self.config.TARGET_CIRCUITS}")

        for i, arxiv_id in enumerate(self.paper_list, start=1):
            if self.paper_counts.get(arxiv_id) is not None:
                continue

            print("\\n" + "=" * 80)
            print(f"Processing paper {i}/{len(self.paper_list)}: {arxiv_id}")
            print(f"Quantum circuits found so far: {total_circuits}/{self.config.TARGET_CIRCUITS}")
            print("=" * 80)

            result = self.process_paper(arxiv_id)
            self.paper_results.append(result)
            self.paper_counts[arxiv_id] = result.quantum_circuits_found
            total_circuits += result.quantum_circuits_found

            print(f"\\n[Paper {i}] Status: {result.status}")
            print(f"[Paper {i}] Figures extracted: {result.total_figures_extracted}")
            print(f"[Paper {i}] Quantum circuits: {result.quantum_circuits_found}")

            if total_circuits >= self.config.TARGET_CIRCUITS:
                print("\\n" + "=" * 80)
                print(f"TARGET REACHED: {total_circuits}")
                print("=" * 80)
                break

            if i < len(self.paper_list) and total_circuits < self.config.TARGET_CIRCUITS:
                print(f"\\nWaiting {self.config.SLEEP_BETWEEN_PAPERS} seconds...\\n")
                self._write_outputs()
                time.sleep(self.config.SLEEP_BETWEEN_PAPERS)

        self._write_outputs()
        self._print_summary()

    def process_paper(self, arxiv_id: str) -> PaperProcessingResult:
        """Process a single arXiv paper to find quantum circuits."""
        pdf_path = self.pdf_processor.download_pdf(arxiv_id)
        if not pdf_path:
            return PaperProcessingResult(arxiv_id, None, "download_failed", 0, 0, "Failed to download PDF")

        try:
            print(f"[{arxiv_id}] Converting PDF with Docling...")
            conv_res = self.pdf_processor.doc_converter.convert(str(pdf_path))
            doc = conv_res.document

            figures = self.pdf_processor.extract_figures(doc, arxiv_id, str(pdf_path))
            print(f"[{arxiv_id}] Extracted {len(figures)} figures")

            quantum_found = 0
            for fig in figures:
                if len(self.quantum_circuits) >= self.config.TARGET_CIRCUITS:
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
                        print(f"  [ERROR] Classifier failed: {e}")
                        is_circuit, gates, problem, confidence = False, [], "", "ERROR"
                else:
                    is_circuit, gates, problem, confidence = self.classifier.classify(
                        fig["caption"], arxiv_id, fig["figure_number"]
                    )

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

                if is_circuit:
                    self._add_to_dataset(fig, gates, problem)
                    quantum_found += 1
                    print(f"  ✓ Figure {fig['figure_number']}: QUANTUM CIRCUIT")
                else:
                    print(f"  ✗ Figure {fig['figure_number']}: Not a circuit")

            return PaperProcessingResult(arxiv_id, str(pdf_path), "ok", len(figures), quantum_found)

        except Exception as e:
            print(f"[{arxiv_id}] ERROR: {e}")
            return PaperProcessingResult(arxiv_id, str(pdf_path), "processing_error", 0, 0, str(e))

    def _add_to_dataset(self, fig: Dict[str, Any], gates: List[str], problem: str):
        """Add a figure to the dataset."""
        safe_id = fig["arxiv_id"].replace("/", "_")
        filename = f"{safe_id}_p{fig['page_number']:02d}_f{fig['pic_index']:02d}.png"
        
        dest_path = self.config.DATASET_IMAGES_DIR / filename
        try:
            shutil.copy(fig["image_path"], dest_path)
        except Exception as e:
            print(f"  [WARN] Failed to copy image: {e}")
            return

        entry = QuantumCircuitEntry(
            image_filename=filename,
            arxiv_id=fig["arxiv_id"],
            page_number=fig["page_number"],
            figure_number=fig["figure_number"],
            quantum_gates=gates,
            quantum_problem=problem,
            descriptions=[fig["caption"]],
            text_positions=[]
        )
        self.quantum_circuits[filename] = entry

    def _read_paper_list(self) -> List[str]:
        """Read the list of arXiv IDs from file."""
        if not self.config.PAPER_LIST_FILE.exists():
            print(f"[ERROR] Paper list file not found: {self.config.PAPER_LIST_FILE}")
            return []
        
        with open(self.config.PAPER_LIST_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _write_dataset_json(self):
        """Write the dataset JSON file."""
        data = {fname: asdict(entry) for fname, entry in self.quantum_circuits.items()}
        with open(self.config.DATASET_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _write_outputs(self):
        """Write all output files."""
        self._write_dataset_json()
        
        # Write CSV
        with open(self.config.PAPER_COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arxiv_id", "quantum_circuits_found"])
            for arxiv_id, count in self.paper_counts.items():
                if count is not None:
                    writer.writerow([arxiv_id, count])
        
        # Write classifications
        with open(self.config.ALL_CLASSIFICATIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.all_classifications, f, indent=2)

    def _print_summary(self):
        """Print final summary."""
        print("\\n" + "=" * 80)
        print("DATASET BUILDING COMPLETE")
        print("=" * 80)
        print(f"Total quantum circuits collected: {len(self.quantum_circuits)}")
        print(f"Dataset directory: {self.config.DATASET_DIR}")
        print(f"Images directory: {self.config.DATASET_IMAGES_DIR}")
        print("=" * 80)
