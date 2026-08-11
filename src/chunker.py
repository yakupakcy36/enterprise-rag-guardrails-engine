"""Document chunking with sentence boundary preservation and metadata tracking."""

import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count_approx: int


class TextChunker:
    """Sliding-window chunker preserving sentence and paragraph boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, min_chunk_len: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def chunk_document(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Split raw text into clean overlapping chunks."""
        cleaned = re.sub(r"\r\n|\r", "\n", text).strip()
        if not cleaned:
            return []

        # Split into sentences or paragraph blocks
        sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
        chunks: List[DocumentChunk] = []

        current_text = ""
        start_char = 0
        chunk_idx = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            if len(current_text) + len(sentence) <= self.chunk_size:
                current_text = f"{current_text} {sentence}".strip() if current_text else sentence
            else:
                if len(current_text) >= self.min_chunk_len:
                    end_char = start_char + len(current_text)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_c{chunk_idx}",
                        doc_id=doc_id,
                        text=current_text,
                        start_char=start_char,
                        end_char=end_char,
                        token_count_approx=len(current_text.split()),
                    ))
                    chunk_idx += 1

                # Overlap: keep the trailing portion of the previous chunk
                overlap_len = min(len(current_text), self.chunk_overlap)
                overlap_text = current_text[-overlap_len:] if overlap_len > 0 else ""
                start_char += max(1, len(current_text) - overlap_len)
                current_text = f"{overlap_text} {sentence}".strip()

        # Final trailing chunk
        if len(current_text) >= self.min_chunk_len:
            end_char = start_char + len(current_text)
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                text=current_text,
                start_char=start_char,
                end_char=end_char,
                token_count_approx=len(current_text.split()),
            ))

        return chunks
