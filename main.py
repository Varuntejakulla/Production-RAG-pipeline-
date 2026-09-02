from indexing.document_loader import DocumentLoader
from indexing.document_cleaning import DocumentCleaning
from indexing.document_chunking import DocumentChunker
from indexing.document_embedder import DocumentEmbeder

document_loader = DocumentLoader()
documents = document_loader.load(
    "./document-Rag-pipe-line.md"
)
print("loading done ..")
document_cleaning = DocumentCleaning()
cleaned_documents = document_cleaning.cleaning(documents=documents)
print("cleaning done ...")
document_chunking = DocumentChunker()
chunking_preocesss = document_chunking.split(documents=cleaned_documents)  
print("chunking is also done ...")
document_emebediing = DocumentEmbeder()
docum_embedibngs=document_emebediing.embeding_documents(chunking_preocesss)




