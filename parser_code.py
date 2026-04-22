"""
Resume Parser Module
====================
Extracts structured text from resume PDFs using PyMuPDF (fitz),
preserving section hierarchy, font metadata, and chronological order.

Handles: single-column, two-column layouts, and missing sections.
"""

import re
import argparse
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Section detection patterns
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "SUMMARY": [
        r"(?i)\b(summary|profile|objective|about\s*me|professional\s*summary|career\s*summary)\b"
    ],
    "EXPERIENCE": [
        r"(?i)\b(experience|work\s*experience|employment|professional\s*experience|work\s*history)\b"
    ],
    "EDUCATION": [
        r"(?i)\b(education|academic|qualifications|degrees?)\b"
    ],
    "SKILLS": [
        r"(?i)\b(skills|technical\s*skills|technologies|competencies|expertise|tools)\b"
    ],
    "CERTIFICATIONS": [
        r"(?i)\b(certifications?|licenses?|credentials|accreditations?|certificates?)\b"
    ],
    "PROJECTS": [
        r"(?i)\b(projects?|portfolio|personal\s*projects?|key\s*projects?)\b"
    ],
}

# Patterns for extracting structured experience entries
DURATION_PATTERN = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s.,]*\d{2,4})"
    r"[\s\-–—to]+"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s.,]*\d{2,4}|[Pp]resent|[Cc]urrent)",
    re.IGNORECASE,
)

BULLET_PATTERN = re.compile(r"^\s*[•●○▪▸►\-–—\*]\s*", re.MULTILINE)


class ResumeParser:
    """
    Parses resume PDFs into structured dictionaries and LangChain Document
    chunks suitable for RAG indexing.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Extract raw text block-by-block from a PDF, detect sections
        using font-size heuristics + regex, and return a structured dict.

        Returns
        -------
        dict with keys:
            raw_text : str – full extracted text
            sections : dict – { SECTION_NAME: str | list[dict] }
            metadata : dict – { page_count, fonts_used, candidate_name }
        """
        doc = fitz.open(pdf_path)
        blocks = self._extract_blocks(doc)
        raw_text = "\n".join(b["text"] for b in blocks)
        sections = self._detect_sections(blocks)
        metadata = self._extract_metadata(doc, blocks, raw_text)
        doc.close()

        return {
            "raw_text": raw_text,
            "sections": sections,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # RAG chunking
    # ------------------------------------------------------------------

    def chunk_for_rag(self, parsed: dict) -> list[Document]:
        """
        Split parsed resume sections into LangChain Document objects
        with metadata {section, company, date_range}.
        """
        documents: list[Document] = []

        for section_name, content in parsed.get("sections", {}).items():
            if section_name == "EXPERIENCE" and isinstance(content, list):
                # Each experience entry gets its own chunks
                for entry in content:
                    entry_text = self._format_experience_entry(entry)
                    chunks = self.text_splitter.split_text(entry_text)
                    for chunk in chunks:
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "section": section_name,
                                    "company": entry.get("company", "Unknown"),
                                    "date_range": entry.get("duration", ""),
                                },
                            )
                        )
            elif isinstance(content, str) and content.strip():
                chunks = self.text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "section": section_name,
                                "company": "",
                                "date_range": "",
                            },
                        )
                    )

        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_blocks(self, doc: fitz.Document) -> list[dict]:
        """
        Extract text blocks from every page, sorted by reading order.
        For two-column layouts, sorts by x-coordinate first (left-to-right),
        then by y-coordinate (top-to-bottom).
        """
        all_blocks: list[dict] = []
        fonts_seen: set[float] = set()

        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_width = page.rect.width

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # skip image blocks
                    continue

                block_text_parts: list[str] = []
                max_font_size = 0.0
                font_flags = 0

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 12.0)
                        max_font_size = max(max_font_size, font_size)
                        font_flags |= span.get("flags", 0)
                        fonts_seen.add(font_size)
                    block_text_parts.append(line_text)

                text = "\n".join(block_text_parts).strip()
                if not text:
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                x0, y0 = bbox[0], bbox[1]

                # Determine column: left half vs right half (for 2-col layouts)
                column = 0 if x0 < page_width / 2 else 1

                all_blocks.append(
                    {
                        "text": text,
                        "font_size": max_font_size,
                        "font_flags": font_flags,
                        "page": page_num,
                        "x": x0,
                        "y": y0,
                        "column": column,
                        "is_bold": bool(font_flags & 2**4),  # bit 4 = bold
                    }
                )

        # Sort: page → column (left first) → y position
        all_blocks.sort(key=lambda b: (b["page"], b["column"], b["y"]))
        return all_blocks

    def _detect_sections(self, blocks: list[dict]) -> dict:
        """
        Identify section boundaries using font-size heuristics and regex.
        Returns a dict mapping section names to their content.
        """
        if not blocks:
            return {}

        # Compute median font size to detect "headers" (larger than median)
        font_sizes = [b["font_size"] for b in blocks]
        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 12.0

        # Label each block
        labeled_blocks: list[tuple[Optional[str], dict]] = []
        for block in blocks:
            section_label = self._match_section(block["text"])
            is_heading = (
                block["font_size"] > median_size * 1.1
                or block["is_bold"]
            )
            if section_label and is_heading:
                labeled_blocks.append((section_label, block))
            elif section_label:
                # Regex matched but font isn't larger — still accept if
                # the block text is short (likely a header line)
                if len(block["text"].strip()) < 60:
                    labeled_blocks.append((section_label, block))
                else:
                    labeled_blocks.append((None, block))
            else:
                labeled_blocks.append((None, block))

        # Group blocks under their detected section
        sections: dict[str, list[str]] = {}
        current_section: Optional[str] = None

        for label, block in labeled_blocks:
            if label is not None:
                current_section = label
                if current_section not in sections:
                    sections[current_section] = []
                # Don't add the header text itself to content
            elif current_section:
                sections[current_section].append(block["text"])
            # else: content before any section header — skip or assign to SUMMARY
            else:
                # Pre-header content often is the name / contact info
                pass

        # Post-process: parse EXPERIENCE into structured entries
        result: dict = {}
        for section, text_blocks in sections.items():
            joined = "\n".join(text_blocks)
            if section == "EXPERIENCE":
                result[section] = self._parse_experience(joined)
            else:
                result[section] = joined.strip()

        return result

    def _match_section(self, text: str) -> Optional[str]:
        """Match block text against known section header patterns."""
        clean = text.strip()
        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, clean):
                    return section_name
        return None

    def _parse_experience(self, text: str) -> list[dict]:
        """
        Parse experience section text into structured entries:
        [{company, title, duration, bullets: [str]}]
        """
        entries: list[dict] = []
        lines = text.split("\n")

        current_entry: Optional[dict] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line contains a date range (signals new entry)
            duration_match = DURATION_PATTERN.search(stripped)

            # Detect bullet points
            is_bullet = bool(BULLET_PATTERN.match(stripped))

            if duration_match and not is_bullet:
                # New experience entry
                if current_entry:
                    entries.append(current_entry)

                duration = duration_match.group(0).strip()
                # Remove duration from line to get title/company
                remaining = stripped.replace(duration, "").strip(" \t|–—-,")

                # Heuristic: try to split "Title at Company" or "Title, Company"
                title, company = self._split_title_company(remaining)

                current_entry = {
                    "company": company,
                    "title": title,
                    "duration": duration,
                    "bullets": [],
                }
            elif is_bullet and current_entry:
                bullet_text = BULLET_PATTERN.sub("", stripped).strip()
                if bullet_text:
                    current_entry["bullets"].append(bullet_text)
            elif current_entry:
                # Non-bullet continuation — might be company/title on separate line
                if not current_entry["company"] or current_entry["company"] == "Unknown":
                    current_entry["company"] = stripped
                elif not current_entry["title"]:
                    current_entry["title"] = stripped
                else:
                    # Treat as a bullet without a marker
                    current_entry["bullets"].append(stripped)
            else:
                # First lines before any date — could be title/company
                current_entry = {
                    "company": stripped,
                    "title": "",
                    "duration": "",
                    "bullets": [],
                }

        if current_entry:
            entries.append(current_entry)

        return entries

    def _split_title_company(self, text: str) -> tuple[str, str]:
        """Attempt to split 'Software Engineer at Google' into (title, company)."""
        separators = [" at ", " @ ", " - ", ", ", " | "]
        for sep in separators:
            if sep in text:
                parts = text.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return text.strip(), "Unknown"

    def _extract_metadata(
        self, doc: fitz.Document, blocks: list[dict], raw_text: str
    ) -> dict:
        """Extract document-level metadata."""
        # Candidate name heuristic: first block with largest font on page 1
        candidate_name = "Unknown"
        if blocks:
            page0_blocks = [b for b in blocks if b["page"] == 0]
            if page0_blocks:
                largest = max(page0_blocks, key=lambda b: b["font_size"])
                candidate_name = largest["text"].strip().split("\n")[0]

        fonts_used = sorted(set(b["font_size"] for b in blocks))

        return {
            "page_count": len(doc),
            "fonts_used": fonts_used,
            "candidate_name": candidate_name,
            "char_count": len(raw_text),
        }

    def _format_experience_entry(self, entry: dict) -> str:
        """Format a structured experience entry back into text for chunking."""
        parts = []
        if entry.get("title"):
            parts.append(entry["title"])
        if entry.get("company"):
            parts.append(f"at {entry['company']}")
        if entry.get("duration"):
            parts.append(f"({entry['duration']})")
        header = " ".join(parts)

        bullets = "\n".join(f"• {b}" for b in entry.get("bullets", []))
        return f"{header}\n{bullets}" if bullets else header


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a resume PDF into structured data."
    )
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument(
        "--chunks", action="store_true", help="Also show RAG chunks"
    )
    args = parser.parse_args()

    rp = ResumeParser()
    result = rp.parse_pdf(args.pdf_path)

    print("=" * 60)
    print("RESUME PARSE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    if args.chunks:
        chunks = rp.chunk_for_rag(result)
        print("\n" + "=" * 60)
        print(f"RAG CHUNKS ({len(chunks)} total)")
        print("=" * 60)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1} [{chunk.metadata}] ---")
            print(chunk.page_content)
