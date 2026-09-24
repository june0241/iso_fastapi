from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Odoo ISO Reference Embedding Service"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None  # If using full URL e.g. Qdrant Cloud or remote endpoint
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "odoo_iso_references"

    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_VECTOR_SIZE: int = 384

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
