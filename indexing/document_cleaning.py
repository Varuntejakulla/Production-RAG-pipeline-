import re
from langchain_core.documents import Document
from common.file_type_dection import FileTypeDetector
import unicodedata

class DocumentCleaning:

    def __init__(self):
        self.file_type = FileTypeDetector()

    def cleaning(self,documents:list[Document])->list[Document]:

        cleaned_documents = []

        for document in documents:
            cleaned_text= self._clean_text(
                document.page_content
            )

            if not cleaned_text:
                continue
            cleaned_document = Document(
                page_content=cleaned_text,
                metadata= document.metadata.copy()
            )
            cleaned_documents.append(
                cleaned_document
            )
            return cleaned_documents
        
        
    def _clean_text(
        self,
        text: str
    ) -> str:

        # 1. Normalize Unicode
        text = unicodedata.normalize(
            "NFKC",
            text
        )

        # 2. Normalize line endings
        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        # 3. Remove spaces at line endings
        text = re.sub(
            r"[ \t]+$",
            "",
            text,
            flags=re.MULTILINE
        )

        # 4. Replace multiple spaces/tabs
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # 5. Limit excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        # 6. Remove leading/trailing whitespace
        text = text.strip()

        return text
        

