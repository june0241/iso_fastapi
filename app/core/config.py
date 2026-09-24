from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Odoo ISO Reference Embedding Service"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 14251
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

    # LLM Settings (Ollama with Gemma3:12b)
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "gemma3:12b"
    EXTRACTION_PASSES: int = 1
    LLM_TEMPERATURE: float = 0.1

    # PDF Layout & Windowing Settings
    MIN_PAGE_TEXT_CHARS: int = 40
    IMAGE_AREA_RATIO: float = 0.80
    OCR_DPI: int = 200
    MAX_WINDOW_CHARS: int = 1800

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
