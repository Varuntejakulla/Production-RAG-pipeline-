from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings

class Chunker:
    """
    Splits documents into semantically coherent, overlapping chunks.
    Why: LLMs have token limits, and retrieval quality depends on
    chunk granularity. Overlap preserves context across boundaries.
    """

    def __init__(self):
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,                # token-aware if using tiktoken
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,               # preserves position metadata
        )
        
    def split(self, documents: List[Document]) -> List[Document]:
        """
        RecursiveCharacterTextSplitter tries separators hierarchically:
        1. Paragraph breaks (\n\n) — best semantic boundaries
        2. Line breaks (\n)
        3. Sentence boundaries (. )
        4. Word boundaries (space)
        5. Character-level as last resort

        This hierarchy ensures chunks break at natural boundaries
        whenever possible, producing higher-quality embeddings.
        """
        chunks = self._splitter.split_documents(documents)

        # Enrich metadata for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks
    