"""
Quantum Circuit Metadata Generator
=================================

Generates comprehensive JSON metadata for quantum circuit images.

Output structure per image (top-level key = image filename):

{
    "<image_filename>": {
        "arxiv_id": str,
        "page_number": int,           # 1-based PDF page number
        "figure_number": int,         # -1, -2, ... if no figure number
        "quantum_gates": [str, ...],  # may be empty list if none detected
        "quantum_problem": str,       # empty string if not identified
        "descriptions": [str, ...],   # caption/context texts ("Fig." prefix removed)
        "text_positions": [
                [start, end],              # 0-based character offsets into the
                ...                        # string returned by pymupdf.Page.get_text()
        ]
    }
}

Text position meaning
---------------------

For each description string stored in "descriptions", we search the **full PDF text** 
(extracted by PyMuPDF page-by-page and concatenated).
We define a "token" as a whitespace-separated word.
The position is specificed as a tuple ``(start_token_index, end_token_index)`` 
referring to the **global** token count in the entire document.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pymupdf
import re
import torch
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class QuantumCircuitMetadataGenerator:
    """Generate metadata for quantum circuit images"""
    
    # Quantum gate keywords
    GATE_NAMES = [
        # Standard Single/Double
        "H", "Hadamard", "X", "Y", "Z", "Pauli",
        "CNOT", "CX", "CZ", "SWAP", "Toffoli", "CCZ",
        # Rotations
        "RX", "RY", "RZ", "U", "U1", "U2", "U3",
        "S", "T", "P", "Ph", "Phase", "SX", "√X", "√NOT",
        # Multi-Qubit / Advanced
        "Fredkin", "CSWAP", "CCX", "iSWAP",
        "XX", "YY", "ZZ", "RXX", "RYY", "RZZ",
        "Mølmer-Sørensen", "MS", "Mølmer", "Sørensen",
        "SYC", "Sycamore", "Fsim", "Givens",
        # Measurement
        "Measurement", "M", "Measure"
    ]
    
    # Detailed mapping: Keyword -> Canonical Problem Name
    # Longest/most specific keywords should be matched first in logic, or we use finding logic.
    PROBLEM_KEYWORDS = {
        # --- Core Algorithms ---
        "shor": "Shor's Algorithm",
        "factorization": "Shor's Algorithm",
        "prime factors": "Shor's Algorithm",
        "grover": "Grover's Algorithm",
        "unstructured search": "Grover's Algorithm",
        "amplitude amplification": "Grover's Algorithm",
        "deutsch": "Deutsch-Jozsa Algorithm",
        "simon": "Simon's Algorithm",
        "bernstein-vazirani": "Bernstein-Vazirani Algorithm",
        "bernstein vazirani": "Bernstein-Vazirani Algorithm",
        "phase estimation": "Quantum Phase Estimation",
        "qpe": "Quantum Phase Estimation",
        "eigenphase": "Quantum Phase Estimation",
        "fourier transform": "Quantum Fourier Transform",
        "qft": "Quantum Fourier Transform",
        "hhl": "HHL Algorithm",
        "linear systems": "HHL Algorithm",
        
        # --- Variational / Optimization ---
        "vqe": "Variational Quantum Eigensolver",
        "variational quantum eigensolver": "Variational Quantum Eigensolver",
        "ground state energy": "Variational Quantum Eigensolver",
        "qaoa": "QAOA",
        "approximate optimization": "QAOA",
        "maxcut": "QAOA",
        "combinatorial optimization": "QAOA",
        "variational classifier": "Variational Quantum Classifier",
        "qnn": "Quantum Neural Network",
        "quantum neural network": "Quantum Neural Network",
        "quantum machine learning": "Quantum Machine Learning",
        "qml": "Quantum Machine Learning",
        "classifier": "Quantum Machine Learning",
        "variational circuit": "Variational Quantum Circuit",
        "parameterized circuit": "Variational Quantum Circuit",
        "ansatz": "Variational Ansatz",
        "ansätze": "Variational Ansatz",
        
        # --- Error Correction ---
        "surface code": "Surface Code",
        "toric code": "Toric Code",
        "stabilizer": "Quantum Error Correction",
        "error correction": "Quantum Error Correction",
        "qec": "Quantum Error Correction",
        "repetition code": "Quantum Error Correction",
        "fault toleran": "Fault Tolerance",
        "magic state": "Magic State Distillation",
        "logical qubit": "Quantum Error Correction",
        "syndrome measurement": "Quantum Error Correction",
        
        # --- Simulation / Dynamics ---
        "hamiltonian simulation": "Hamiltonian Simulation",
        "trotter": "Trotterization",
        "time evolution": "Quantum Simulation",
        "dynamics": "Quantum Simulation",
        "ising model": "Ising Model Simulation",
        "heisenberg": "Heisenberg Model Simulation",
        "hubbard": "Fermi-Hubbard Simulation",
        
        # --- Communication / States ---
        "teleportation": "Quantum Teleportation",
        "superdense coding": "Superdense Coding",
        "bell state": "Bell State Preparation",
        "ghz": "GHZ State Preparation",
        "greenberger": "GHZ State Preparation",
        "w state": "W State Preparation",
        "entanglement swapping": "Entanglement Swapping",
        
        # --- Hardware / Architecture specific ---
        "randomized benchmarking": "Randomized Benchmarking",
        "state tomography": "Quantum State Tomography",
        "process tomography": "Quantum Process Tomography",
        "readout error": "Readout Error Mitigation",
        "error mitigation": "Error Mitigation",
        "dynamical decoupling": "Dynamical Decoupling",
        "spin echo": "Dynamical Decoupling",
        "rydberg": "Rydberg Atom Array",
        "blockade": "Rydberg Blockade",
        "transmon": "Superconducting Qubit",
        "ion trap": "Trapped Ion",
        "photon": "Photonic Quantum Computing",
        "boson sampling": "Boson Sampling",
        
        # --- Compilation ---
        "compilation": "Circuit Compilation",
        "transpilation": "Circuit Compilation",
        "decomposition": "Gate Decomposition",
        "synthesis": "Circuit Synthesis",
        "optimization": "Circuit Optimization" 
    }

    def __init__(self, ocr_extractor=None, vlm_judge=None):
        """
        Initialize the Metadata Generator.

        Parameters
        ----------
        ocr_extractor : OCRJudge, optional
            Instance of OCRJudge to reuse loaded OCR models.
        vlm_judge : Any, optional
            Instance of a VLM judge (unused placeholder).
        """
        self.ocr = ocr_extractor
        self.vlm = vlm_judge

        self.embedding_model = None
        self.problem_embeddings = None
        self.canonical_names = sorted(list(set(self.PROBLEM_KEYWORDS.values())))
        
        if ST_AVAILABLE:
            try:
                print("  [Metadata] Loading embedding model for problem ID...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Pre-compute embeddings for canonical names
                self.problem_embeddings = self.embedding_model.encode(self.canonical_names, convert_to_tensor=True)
                print("  [Metadata] Model loaded.")
            except Exception as e:
                print(f"  [Metadata] Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def extract_gates_from_image(self, image_path: str) -> List[str]:
        """
        Extract quantum gate names from circuit image using OCR.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        List[str]
            List of detected gate names (e.g., ["H", "CNOT"]).
        """
        gates_found = []
        
        if self.ocr is None or self.ocr.ocr is None:
            # Try harder - don't give up
            print("  [Metadata] WARNING: OCR not available for gate extraction")
            return []  # Empty list, not "Unknown"
        
        try:
            # Run OCR on image
            result = self.ocr.ocr.readtext(image_path)
            
            if not result:
                return []  # No text found - real result, not placeholder
            
            # Extract all text
            all_text = " ".join([detection[1] for detection in result])
            
            # Find gate names
            for gate in self.GATE_NAMES:
                # Case-insensitive search
                pattern = r'\b' + re.escape(gate) + r'\b'
                if re.search(pattern, all_text, re.IGNORECASE):
                    # Normalize gate name
                    if gate.lower() in ["hadamard"]:
                        gates_found.append("H")
                    elif gate.lower() in ["pauli"]:
                        continue  # Skip generic "Pauli", wait for X/Y/Z
                    elif gate.lower() in ["cx"]:
                        gates_found.append("CNOT")
                    elif gate.lower() in ["m", "measure"]:
                        gates_found.append("Measurement")
                    else:
                        gates_found.append(gate.upper())
            
            # Remove duplicates, preserve order
            gates_found = list(dict.fromkeys(gates_found))
            
        except Exception as e:
            print(f"  [Metadata] Gate extraction error: {e}")
            return []  # Error = no data, not "Unknown"
        
        return gates_found  # Could be empty list - that's OK!

    def extract_gates_from_text(self, text: str) -> List[str]:
        """
        Extract quantum gate names from description/context text using Regex.

        Parameters
        ----------
        text : str
            The input text (caption or context).

        Returns
        -------
        List[str]
            List of detected gate names.
        """
        gates_found = []
        if not text:
            return []

        for gate in self.GATE_NAMES:
            # Case-insensitive search with word boundaries
            pattern = r'\b' + re.escape(gate) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize gate name
                if gate.lower() in ["hadamard"]:
                    gates_found.append("H")
                elif gate.lower() in ["pauli"]:
                    continue  # Skip generic "Pauli"
                elif gate.lower() in ["cx"]:
                    gates_found.append("CNOT")
                elif gate.lower() in ["m", "measure", "measurement"]:
                    gates_found.append("Measurement")
                elif gate.lower() in ["ccz"]:
                    gates_found.append("CCZ")
                else:
                    gates_found.append(gate.upper())
        
        return list(dict.fromkeys(gates_found))

    def extract_gates_by_embedding(self, text: str) -> List[str]:
        """
        Embedding-based gate detection from caption/context text.

        Uses sentence-transformers to match text to a curated set of gate phrases.
        Returns canonical gate names when similarity exceeds a threshold.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        List[str]
            List of detected gate names.
        """
        if not ST_AVAILABLE or not text or len(text) < 5:
            return []

        try:
            phrases_map = {
                "Hadamard gate": "H",
                "Pauli-X gate": "X",
                "Pauli-Y gate": "Y",
                "Pauli-Z gate": "Z",
                "CNOT gate": "CNOT",
                "controlled-NOT": "CNOT",
                "controlled-Z gate": "CZ",
                "CZ gate": "CZ",
                "SWAP gate": "SWAP",
                "Toffoli gate": "Toffoli",
                "CCZ gate": "CCZ",
                "RX rotation": "RX",
                "RY rotation": "RY",
                "RZ rotation": "RZ",
                "U3 gate": "U3",
                "U2 gate": "U2",
                "U1 gate": "U1",
                "S gate": "S",
                "T gate": "T",
                "Measurement": "Measurement",
            }

            phrases = list(phrases_map.keys())
            model = self.embedding_model
            if model is None:
                return []

            text_emb = model.encode(text[:1000], convert_to_tensor=True)
            phrase_embs = model.encode(phrases, convert_to_tensor=True)
            scores = util.cos_sim(text_emb, phrase_embs)[0]

            found: List[str] = []
            for i, sc in enumerate(scores):
                if sc.item() > 0.20:
                    found.append(phrases_map[phrases[i]])

            # Deduplicate while preserving order
            return list(dict.fromkeys(found))
        except Exception:
            return []
    
    def identify_quantum_problem(self, caption: str, context_text: str) -> str:
        """
        Identify the quantum problem or algorithm associated with the circuit.

        Priority:
        1. Caption text keyword match.
        2. Context text keyword match.
        3. Embedding-based similarity match (combined text).
        4. Fallback -> "Quantum Circuit".

        Parameters
        ----------
        caption : str
            Figure caption.
        context_text : str
            Surrounding context text.

        Returns
        -------
        str
            Canonical name of the problem (e.g., "Shor's Algorithm") or "Quantum Circuit".
        """

        def _clean_text(t: str) -> str:
            t = (t or "").lower()
            return t.replace("¨ ", "").replace("¨", "")

        def _keyword_match(text: str) -> str:
            if not text:
                return ""
            best = ""
            best_len = 0
            for keyword, canonical_name in self.PROBLEM_KEYWORDS.items():
                if keyword in text:
                    if len(keyword) > best_len:
                        best_len = len(keyword)
                        best = canonical_name
            return best

        cap_clean = _clean_text(caption)
        ctx_clean = _clean_text(context_text)

        # 1) Caption-only keywords
        best = _keyword_match(cap_clean)
        if best:
            return best

        # 2) Context-only keywords (figure mentions + surrounding text)
        best = _keyword_match(ctx_clean)
        if best:
            return best

        # 3) Embedding-based fallback on combined text
        combined = (cap_clean + " " + ctx_clean).strip()
        if self.embedding_model and self.problem_embeddings is not None and len(combined) > 10:
            try:
                query_embedding = self.embedding_model.encode(combined[:1000], convert_to_tensor=True)
                cos_scores = util.cos_sim(query_embedding, self.problem_embeddings)[0]
                top_result = torch.topk(cos_scores, k=1)
                score = top_result.values.item()
                idx = top_result.indices.item()
                if score > 0.15:
                    return self.canonical_names[idx]
            except Exception:
                pass

        # 4) Fallback
        return "Quantum Circuit"

    
        return positions  # Could be shorter than descriptions list - that's OK!
    
    def extract_text_positions(self, pdf_path: str, page_num: int, descriptions: List[str]) -> List[Tuple[int, int]]:
        """
        Extract GLOBAL token-based start/end positions of descriptions in PDF.

        Scans the ENTIRE PDF to find the description.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF.
        page_num : int
            (Unused) Page number hint.
        descriptions : List[str]
            List of unique description strings to locate.

        Returns
        -------
        List[Tuple[int, int]]
            List of (start_token_idx, end_token_idx) tuples relative to the full document token sequence.
        """
        positions = []
        
        if not pdf_path or not descriptions:
            return []
        
        try:
            doc = pymupdf.open(pdf_path)
            
            # 1. Build Global Token Sequence
            global_tokens = []
            
            # Basic normalization helper
            def normalize_token(t):
                # Remove punctuation, lower case
                t = t.lower()
                t = re.sub(r'[^\w\s]', '', t) 
                return t

            for p_idx in range(len(doc)):
                page_text = doc[p_idx].get_text()
                # Use split() for consistent whitespace tokenization
                page_tokens = page_text.split() 
                global_tokens.extend(page_tokens)
            
            doc.close()
            
            # Pre-calculate normalized global tokens for matching
            global_tokens_norm = [normalize_token(t) for t in global_tokens]
            n_global = len(global_tokens)

            # 2. Search for descriptions
            for desc in descriptions:
                if not desc: 
                    continue
                
                # Tokenize the description using same split
                desc_tokens = desc.split()
                if not desc_tokens:
                    continue
                    
                desc_tokens_norm = [normalize_token(t) for t in desc_tokens]
                n_desc = len(desc_tokens_norm)
                
                # Heuristic: If description is huge, take the first 20 tokens to find the start
                # This helps if the end is cut off or slightly different
                search_len = min(n_desc, 50) 
                search_seq = desc_tokens_norm[:search_len]
                
                found_idx = -1
                
                # ROBUST MATCHING:
                # Instead of requiring 100% match of the full description (which fails on PDF artifacts),
                # we look for a strong "anchor" of the first few meaningful tokens.
                # dynamic fallback strategy: 8 -> 6 -> 4 -> 3
                
                found_anchor_idx = -1
                matched_anchor_len = 0
                
                # Check anchors of decreasing size to handle immediate token mismatches
                for try_len in [8, 6, 4, 3]:
                    if try_len > n_desc:
                        continue
                        
                    anchor_seq = desc_tokens_norm[:try_len]
                    start_token = anchor_seq[0]
                    possible_starts = [idx for idx, t in enumerate(global_tokens_norm) if t == start_token]
                    
                    for idx in possible_starts:
                        if idx + try_len > n_global:
                            continue
                        
                        match = True
                        for k in range(1, try_len):
                             if global_tokens_norm[idx+k] != anchor_seq[k]:
                                 match = False
                                 break
                        
                        if match:
                            found_anchor_idx = idx
                            matched_anchor_len = try_len
                            # Prefer first match for longer anchors, usually correct
                            break
                    
                    if found_anchor_idx != -1:
                        print(f"  [Metadata] Found text anchor with length {try_len}")
                        break
                
                if found_anchor_idx != -1:
                    # found the start!
                    start_global = found_anchor_idx
                    # The end is simply start + length of description (in tokens)
                    # We can try to be smarter and verify the end, but counting length is requested behavior.
                    
                    # Adjust end if we know there was a token mismatch? 
                    # Complex. For now, strict count based on description length is safer default.
                    end_global = found_anchor_idx + n_desc - 1
                    
                    # Clamp to document bounds
                    
                    # Clamp to document bounds just in case
                    end_global = min(end_global, n_global - 1)
                    
                    positions.append((start_global, end_global))
                    print(f"  [Metadata] Found global tokens at {start_global}-{end_global}")
                else:
                    print(f"  [Metadata] Description not found in full document tokens.")
                    
        except Exception as e:
            print(f"  [Metadata] Global token match error: {e}")
            return []
        
        return positions

    def _clean_description(self, text: str) -> str:
        """
        Remove leading "Fig.", "FIG.", "Figure" prefixes from a description.

        Parameters
        ----------
        text : str
            Raw caption text.

        Returns
        -------
        str
            Cleaned caption text.

        Examples
        --------
        - "FIG. 3: ..." -> "..."
        - "Fig. 2. ..." -> "..."
        """
        if not text:
            return ""

        s = text.strip()
        # "remove fig. figur. FIG name in the strting."
        # Aggressive removal of Figure X patterns
        # 1. Matches "FIG. 11", "Figure 2", "Fig 3." followed by anything up to a colon or text start
        # e.g. "Fig. 11 shows..." -> "shows..."
        
        # Regex explanation:
        # ^(fig(ure)?\.?)  : Starts with Fig, Figure, Fig., Figure. (case insensitive)
        # \s*              : optional space
        # \d+              : the number (e.g. 11)
        # [:\.\-]*         : separator chars (:, ., -)
        # \s*              : trailing space
        
        # Remove common figure prefixes with numbers, including supplementary like S1
        pattern_num = r"^(?:fig(?:ure)?\.?|figure)\s*(?:s)?\s*\d+[a-zA-Z0-9]*[:\.\-]*\s*"
        s = re.sub(pattern_num, "", s, flags=re.IGNORECASE)
        # Also remove bare prefixes like "Fig." or "FIG:" if present at start
        pattern_bare = r"^(?:fig(?:ure)?\.?|figure)[:\.\-]*\s*"
        s = re.sub(pattern_bare, "", s, flags=re.IGNORECASE)
        
        return s.strip()
    
    def generate_metadata(self, 
                          image_path: str,
                          arxiv_id: str,
                          page_num: int,
                          fig_num: int,
                          caption: str,
                          context_mentions: List[str],
                          pdf_path: str,
                          vlm_gates: List[str] = None) -> Dict:
        """
        Generate complete metadata for a quantum circuit image.

        Aggregates gates from OCR, Regex, and Embeddings.
        Identifies the problem domain.
        Computes global text positions.

        Parameters
        ----------
        image_path : str
            Path to the circuit image.
        arxiv_id : str
            ArXiv ID.
        page_num : int
            Page number.
        fig_num : int
            Figure number.
        caption : str
            Caption text.
        context_mentions : List[str]
            List of context text snippets.
        pdf_path : str
            Path to the PDF.
        vlm_gates : List[str], optional
            Gates extracted by VLM (unused placeholder).

        Returns
        -------
        Dict
            Dictionary matching the Metadata Schema.
        """
        print(f"  [Metadata] Generating for {arxiv_id} Fig.{fig_num}...")
        
        # Gate detection: caption + OCR (separate from pipeline) + embeddings
        final_gates = set()
        caption_text = caption or ""

        # 1) OCR-based gate extraction from the image (JSON-only)
        ocr_gates = self.extract_gates_from_image(image_path)
        final_gates.update(ocr_gates)
        print(f"  [Metadata] Gates from image OCR: {len(ocr_gates)}")

        # 2) Caption regex
        text_gates = self.extract_gates_from_text(caption_text)
        final_gates.update(text_gates)
        print(f"  [Metadata] Gates from caption regex: {len(text_gates)}")

        # 3) Caption embeddings
        emb_gates = self.extract_gates_by_embedding(caption_text)
        final_gates.update(emb_gates)
        if emb_gates:
            print(f"  [Metadata] Gates from caption embedding: {len(emb_gates)}")
        
        gates = sorted(list(final_gates))
        
        # Identify problem - REAL data only  
        context_text = " ".join(context_mentions) if context_mentions else ""
        problem = self.identify_quantum_problem(caption, context_text)
        
        # Descriptions - use caption primarily. 
        # Do NOT use context_mentions (full page text) as a description.
        descriptions: List[str] = []
        if caption:
            cleaned = self._clean_description(caption)
            if cleaned:
                descriptions = [cleaned]
        # If caption is empty, we have no description. 
        # (Previously we used context_mentions, but that is now full page text).
        # If neither, descriptions stays empty []
        
        # Text positions - REAL positions only
        positions = self.extract_text_positions(pdf_path, page_num, descriptions)
        
        # Build metadata - always include all required fields
        metadata = {
            "arxiv_id": arxiv_id,
            "page_number": int(page_num),
            "figure_number": int(fig_num),
            "quantum_gates": gates or [],
            "quantum_problem": problem or "",
            "descriptions": descriptions or [],
            "text_positions": positions or [],
        }
        
        print(f"  [Metadata] Generated: Gates={len(gates)}, Problem={'Yes' if problem else 'No'}, Desc={len(descriptions)}, Pos={len(positions)}")
        
        return metadata


def build_metadata_from_dataset() -> None:
    """
    Build a single metadata JSON file for all accepted circuit images.

    Reads the final dataset JSON produced by the main pipeline
    (e.g. ``quantum_circuits_8/dataset_8.json``), re-generates rich
    metadata for each image using :class:`QuantumCircuitMetadataGenerator`,
    and writes a single metadata file (`metadata_full_8.json`).
    """

    try:
        from classifiers.stage1_ocr import OCRJudge
    except ImportError:
         print("[WARN] Could not import OCRJudge from classifiers.stage1_ocr")
         # Define dummy if needed or fail
         OCRJudge = None

    def _clean_descriptions_for_seed(raw_descriptions: List[str]) -> List[str]:
        cleaned: List[str] = []
        for d in raw_descriptions or []:
            if not d:
                continue
            s = d.strip()
            if s:
                cleaned.append(s)
        return cleaned

    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "quantum_circuits_8"
    images_dir = dataset_dir / "images_8"

    dataset_json_path = dataset_dir / "dataset_8.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")

    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset: Dict[str, Any] = json.load(f)

    ocr = OCRJudge()
    meta_gen = QuantumCircuitMetadataGenerator(ocr_extractor=ocr)

    all_metadata: Dict[str, Any] = {}

    for image_filename, entry in dataset.items():
        arxiv_id = entry.get("arxiv_id")
        page_number = int(entry.get("page_number", 1))
        figure_number = int(entry.get("figure_number", -1))

        image_path = images_dir / image_filename

        safe_id = arxiv_id.replace("/", "_") if arxiv_id else ""
        pdf_path = (base_dir / "pdfs" / f"{safe_id}.pdf").as_posix() if safe_id else ""

        raw_descriptions = entry.get("descriptions", [])
        seed_descriptions = _clean_descriptions_for_seed(raw_descriptions)

        if seed_descriptions:
            caption = seed_descriptions[0]
            context_mentions = seed_descriptions[1:]
        else:
            caption = ""
            context_mentions = []

        # Enrich context with figure-mention sentences and their surrounding paragraphs
        # across the entire document.
        if pdf_path:
            try:
                doc = pymupdf.open(pdf_path)
                fig_pattern = re.compile(rf"fig(?:ure)?\.?\s*{figure_number}\b", re.IGNORECASE)
                for p_idx in range(len(doc)):
                    page_text = doc[p_idx].get_text()
                    # Sentences that mention this figure
                    rough_sentences = re.split(r"[\n\.]+", page_text)
                    hit_sentences = [s.strip() for s in rough_sentences if s.strip() and fig_pattern.search(s)]
                    context_mentions.extend(hit_sentences)

                    # Surrounding paragraphs: split by blank lines; include any paragraph
                    # that contains a hit sentence
                    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", page_text) if para.strip()]
                    for para in paragraphs:
                        if any(s in para for s in hit_sentences):
                            context_mentions.append(para)
                doc.close()
            except Exception:
                pass

        vlm_gates = entry.get("quantum_gates") or None

        metadata = meta_gen.generate_metadata(
            image_path=image_path.as_posix(),
            arxiv_id=arxiv_id,
            page_num=page_number,
            fig_num=figure_number,
            caption=caption,
            context_mentions=context_mentions,
            pdf_path=pdf_path,
            vlm_gates=vlm_gates,
        )
        # Reject the image if gates not found (empty list)
        # [FIX] Do NOT reject here. If the classifier accepted it, we keep it, even if Metadata Generator found no gates.
        all_metadata[image_filename] = metadata

    out_path = dataset_dir / "metadata_full_8.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"[Metadata Builder] Wrote metadata for {len(all_metadata)} images -> {out_path}")


if __name__ == "__main__":
    build_metadata_from_dataset()
