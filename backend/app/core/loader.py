from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from langchain_core.documents import Document


class DocumentLoader:
    """
    Loads documents into a uniform Document format.
    Why: Downstream components (chunker, embedder) expect a consistent
    interface regardless of source file type.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @staticmethod
    def load(file_path: str | Path) -> List[Document]:
        """
        Load a single file and return a list of Document objects.
        PDFs return one Document per page; text files return one Document.
        """
        path = Path(file_path)

        if path.suffix.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if path.suffix.lower() == ".pdf":
            return DocumentLoader._load_pdf(path)
        else:
            return DocumentLoader._load_text(path)
    @staticmethod
    def _load_pdf(path: Path) -> List[Document]:
        """
        PyMuPDF (fitz) is used instead of pypdf because it:
        - Correctly handles kerning and layout-preserving spacing
        - Is significantly faster (<1s/file for standard PDFs)
        - Preserves page numbers for citation metadata
        """
        doc:Document = fitz.open(path)
        documents = []

        for page_num, page in enumerate(doc):
            text:str = page.get_text("text")
            if not text.strip():
                continue  # skip empty pages

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_num + 1,          # 1-indexed for humans
                        "file_type": "pdf",
                    },
                )
            )

        doc.close()
        return documents
    
    @staticmethod
    def _load_text(path: Path) -> List[Document]:
        """Load plain text / markdown as a single Document."""
        text = path.read_text(encoding="utf-8")
        return [
            Document(
                page_content=text,
                metadata={"source": str(path), "page": 1, "file_type": path.suffix[1:]},
            )
        ]
    