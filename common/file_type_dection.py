from pathlib import Path

class FileTypeDetector:

    SUPPORTED_FILE_EXTENSIONS= {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "markdown",
        ".docx":"documents"
    }

    def detect_file(self,file_path:Path)->str:
        extension = file_path.suffix.lower()
        file_type = self.SUPPORTED_FILE_EXTENSIONS.get(
            extension
            )
        if not  file_type:
            raise ValueError(
                f"Unsupportted file  type: {extension}"
            )
        return file_type
    
