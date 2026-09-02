from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:

    def __init__(self,
                 chunk_size:int=500,
                 chunk_overlap:int=100
                 ):
        self.text_spilter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self,documents:list[Document])->list[Document]:
        chunks = self.text_spilter.split_documents(
            documents
        )
        for index,chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]=index

        return chunks
    
