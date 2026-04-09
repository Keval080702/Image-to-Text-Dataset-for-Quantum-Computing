"""
PDF Processor Module
====================

Handles PDF downloading and figure extraction using Docling.
"""

from __future__ import annotations
import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem
from config import Config


class PDFProcessor:
    """Handles PDF downloading and figure extraction."""

    def __init__(self, config: Config):
        self.config = config
        
        # Setup Docling converter
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        try:
            pipeline_options.images_scale = float(os.environ.get("DOC_IMAGES_SCALE", self.config.IMAGES_SCALE))
        except Exception:
            pipeline_options.images_scale = float(self.config.IMAGES_SCALE)
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def download_pdf(self, arxiv_id: str) -> Optional[Path]:
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

    @staticmethod
    def clean_description(text: str) -> str:
        """Remove leading 'Fig.', 'FIG.', 'Figure' prefixes from a description."""
        if not text:
            return ""

        s = text.strip()
        pattern_num = r"^(?:fig(?:ure)?\.?|figure)\s*(?:s)?\s*\d+[a-zA-Z0-9]*[:\.\\-]*\s*"
        s = re.sub(pattern_num, "", s, flags=re.IGNORECASE)
        pattern_bare = r"^(?:fig(?:ure)?\.?|figure)[:\.\\-]*\s*"
        s = re.sub(pattern_bare, "", s, flags=re.IGNORECASE)

        return s.strip()

    def extract_figures(self, doc: Any, arxiv_id: str, pdf_path: str) -> List[Dict[str, Any]]:
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
            A list of dictionaries, one per extracted figure.
        """
        figures: List[Dict[str, Any]] = []

        for pic_idx, pic in enumerate(getattr(doc, "pictures", []), start=1):
            if not isinstance(pic, PictureItem):
                continue
            prov_list = getattr(pic, "prov", None)
            if not prov_list:
                continue
            page_number = int(getattr(prov_list[0], "page_no", 1))

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
                "context_mentions": [page_full_text],
            })

        return figures

    def _get_caption_and_context(self, pic: PictureItem, doc: Any, pdf_path: str) -> Tuple[str, str]:
        """Retrieve the caption and surrounding text context for a figure."""
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

        # Get full page text
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
            pdf_doc.close()

            return caption if caption else "Figure", text
        except Exception:
            return caption if caption else "Figure", ""

    @staticmethod
    def _extract_figure_number(caption: str) -> Optional[int]:
        """Extract figure number from caption text."""
        if not caption:
            return None
        match = re.search(r"(?:fig(?:ure)?\.?|fig)\s*(\d+)", caption, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _save_figure_image(self, pic: PictureItem, doc: Any, arxiv_id: str, page_number: int, pic_idx: int) -> Optional[str]:
        """Save figure image to disk."""
        try:
            pil_image = pic.get_image(doc)
            if pil_image is None:
                return None

            safe_id = arxiv_id.replace("/", "_")
            filename = f"{safe_id}_p{page_number:02d}_f{pic_idx:02d}.png"
            out_path = self.config.FIGURES_DIR / filename
            pil_image.save(str(out_path), "PNG")
            return str(out_path)
        except Exception as e:
            print(f"  [WARN] Failed to save figure image: {e}")
            return None
