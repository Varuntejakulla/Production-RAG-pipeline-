from indexing.document_loader import DocumentLoader
from indexing.document_cleaning import DocumentCleaning
from indexing.document_chunking import DocumentChunker
from langchain_core.documents import Document
from typing import Dict


class  IndexingPipeline:

    def __init__(self):
        self.documentloader = DocumentLoader()
        self.documenmtchunking = DocumentChunker()
        self.documentloader = DocumentChunker()




document_loader = DocumentLoader()
documents = document_loader.load(
    "./document-Rag-pipe-line.md"
)

document_cleaning = DocumentCleaning()
cleaned_documents = document_cleaning.cleaning(documents=documents)
document_chunking = DocumentChunker()
chunking_preocesss = document_chunking.split(documents=cleaned_documents)
document_chunking = 
