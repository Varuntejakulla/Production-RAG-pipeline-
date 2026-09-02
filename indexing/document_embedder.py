from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class  DocumentEmbeder:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "intfloat/multilingual-e5-base",
            model_kwargs= {
                "device":"cpu"
            },
            encode_kwargs={
                    "normalize_embeddings": True
            }
        )

    def embeding_documents(
            self,
            documents:list[Document]
     )-> list[list[float]]:
        texts= [
              document.page_content
              for document in documents     
            ]
        embeddings = self.embedding_model.embed_documents(
            texts
        )
        return embeddings
        