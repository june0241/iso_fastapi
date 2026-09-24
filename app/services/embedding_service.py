from typing import List
from fastembed import TextEmbedding
from app.core.config import settings
from app.models.schemas import IsoFactCreate


class EmbeddingService:
    """Service to generate dense vector embeddings using FastEmbed."""

    _instance: "EmbeddingService | None" = None
    _model: TextEmbedding | None = None

    def __init__(self):
        if self._model is None:
            self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance

    @staticmethod
    def format_fact_text_for_embedding(doc: IsoFactCreate) -> str:
        """
        Formats an ISO fact into a structured, highly searchable text block 
        with explicit metadata headings and atomic fact content.
        """
        meta = doc.metadata
        sub_clause_str = f" > {meta.sub_clause}" if meta.sub_clause else ""
        odoo_context_parts = []
        if meta.odoo_module:
            odoo_context_parts.append(f"Module: {meta.odoo_module}")
        if meta.odoo_model:
            odoo_context_parts.append(f"Model: {meta.odoo_model}")
        if meta.odoo_process:
            odoo_context_parts.append(f"Process: {meta.odoo_process}")
        odoo_context_str = " | ".join(odoo_context_parts) if odoo_context_parts else "N/A"

        tags_str = ", ".join(meta.tags) if meta.tags else "N/A"

        structured_text = (
            f"[ISO STANDARD]: {meta.iso_standard}\n"
            f"[CLAUSE]: {meta.clause_number} - {meta.clause_title}{sub_clause_str}\n"
            f"[TOPIC]: {meta.topic}\n"
            f"[ODOO MAPPING]: {odoo_context_str}\n"
            f"[TAGS]: {tags_str}\n"
            f"[FACT SUMMARY]: {doc.fact_summary}\n"
            f"[INDEPENDENT FACT CONTENT]:\n{doc.fact_content}"
        )
        return structured_text.strip()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of plain strings into vector lists."""
        generator = self._model.embed(texts)
        return [list(vec) for vec in generator]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string into a vector."""
        vectors = list(self._model.embed([query]))
        return list(vectors[0])

    def embed_iso_documents(self, documents: List[IsoFactCreate]) -> tuple[List[List[float]], List[str]]:
        """
        Processes a batch of ISO fact documents, returns their formatted texts and vector embeddings.
        """
        formatted_texts = [self.format_fact_text_for_embedding(doc) for doc in documents]
        embeddings = self.embed_texts(formatted_texts)
        return embeddings, formatted_texts


# Global singleton instance
embedding_service = EmbeddingService.get_instance()
