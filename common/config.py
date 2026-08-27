from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):

    file_storage_path:Path = Path("data")

    chunking_size=1000
    chunk_overlap=200

    embedding_model:str =(
        ""
    )


    #vector databse  creditionals 
    


    model_config =SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
