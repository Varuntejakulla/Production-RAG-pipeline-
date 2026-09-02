from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    markdown,
    Docx2txtLoader
)

from common.file_type_dection import FileTypeDetector
from typing  import Callable


class DocumentLoader:

    def __init__(self):
        self.file_type_detctor =(
            FileTypeDetector()
        )
        self.loaders:dict[str, Callable[[str],PyPDFLoader|TextLoader|Docx2txtLoader]]= {
            "pdf":lambda file_path:PyPDFLoader(file_path),
            "text":lambda file_path:TextLoader(
                file_path,
                encoding="utf-8"

            ),
            "markdown":lambda file_path:TextLoader(
                file_path,
                encoding="utf-8",
            ),
            "docx":lambda file_path:Docx2txtLoader(
                file_path,
                
            )


        }
    def load(self,file_path:str):
        file_type =(
            self.file_type_detctor.detect_file(
                file_path
            )
        )
        loader_factory = self.loaders.get(file_type)

        if loader_factory is None:
            raise ValueError(f"Unsupported file :{file_path}")
        

        loader = loader_factory(file_path)

        documents = loader.load()

        return documents

    
        
        