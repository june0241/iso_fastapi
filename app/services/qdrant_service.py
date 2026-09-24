import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.models.schemas import (
    CollectionStatsResponse,
    IsoFactCreate,
    IsoFactMetadata,
    IsoSearchFilter,
    IsoSearchResultItem,
)


class QdrantService:
    """Service to manage Qdrant collections, upserts, and filtered vector search."""

    def __init__(self):
        if settings.QDRANT_URL and settings.QDRANT_URL.strip():
            self.client = QdrantClient(
                url=settings.QDRANT_URL.strip(),
                api_key=settings.QDRANT_API_KEY or None,
                https=settings.QDRANT_URL.strip().startswith("https"),
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                https=False,
                prefer_grpc=False,
                api_key=settings.QDRANT_API_KEY or None,
                check_compatibility=False,
            )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def ensure_collection(self) -> None:
        """Verify if collection exists; create and set up payload indexes if missing."""
        try:
            collections_response = self.client.get_collections()
            existing_names = [c.name for c in collections_response.collections]
            
            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=settings.EMBEDDING_VECTOR_SIZE,
                        distance=rest_models.Distance.COSINE
                    )
                )
                self._setup_payload_indexes()
        except Exception as e:
            print(f"[Warning] Could not initialize Qdrant collection during startup: {e}")

    def _setup_payload_indexes(self) -> None:
        """Create indexes on payload fields for high-performance metadata filtering."""
        fields_to_index = [
            ("metadata.iso_standard", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.clause_number", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.topic", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.odoo_module", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.odoo_model", rest_models.PayloadSchemaType.KEYWORD),
            ("metadata.tags", rest_models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass

    def check_health(self) -> str:
        """Check connection to Qdrant."""
        try:
            self.client.get_collections()
            return "connected"
        except Exception as e:
            return f"disconnected ({str(e)})"

    def upsert_iso_facts(
        self,
        documents: List[IsoFactCreate],
        vectors: List[List[float]],
        formatted_texts: List[str],
    ) -> List[dict]:
        """
        Upsert a batch of ISO facts with vector embeddings and rich metadata payloads.
        """
        self.ensure_collection()
        points: List[rest_models.PointStruct] = []
        ingestion_results = []

        for doc, vector, fmt_text in zip(documents, vectors, formatted_texts):
            # Generate deterministic UUID if fact_id is provided, else standard uuid4
            if doc.fact_id:
                try:
                    point_id = str(uuid.UUID(doc.fact_id))
                except ValueError:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.fact_id))
            else:
                point_id = str(uuid.uuid4())

            payload = {
                "fact_id": doc.fact_id or point_id,
                "fact_summary": doc.fact_summary,
                "fact_content": doc.fact_content,
                "formatted_context": fmt_text,
                "metadata": doc.metadata.model_dump(),
            }

            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

            ingestion_results.append({
                "fact_id": doc.fact_id or point_id,
                "status": "indexed",
                "iso_standard": doc.metadata.iso_standard,
                "clause_number": doc.metadata.clause_number,
            })

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ingestion_results

    def search_iso_facts(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[IsoSearchFilter] = None,
    ) -> List[IsoSearchResultItem]:
        """
        Search ISO collection using semantic similarity and optional payload filters.
        """
        qdrant_filter = None
        if filters:
            must_conditions = []
            if filters.iso_standard:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.iso_standard",
                        match=rest_models.MatchValue(value=filters.iso_standard)
                    )
                )
            if filters.clause_number:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.clause_number",
                        match=rest_models.MatchValue(value=filters.clause_number)
                    )
                )
            if filters.topic:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.topic",
                        match=rest_models.MatchValue(value=filters.topic)
                    )
                )
            if filters.odoo_module:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.odoo_module",
                        match=rest_models.MatchValue(value=filters.odoo_module)
                    )
                )
            if filters.odoo_model:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.odoo_model",
                        match=rest_models.MatchValue(value=filters.odoo_model)
                    )
                )
            if filters.tag:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key="metadata.tags",
                        match=rest_models.MatchValue(value=filters.tag)
                    )
                )

            if must_conditions:
                qdrant_filter = rest_models.Filter(must=must_conditions)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            search_results = response.points
        elif hasattr(self.client, "search"):
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        else:
            search_results = self.client.search_points(
                collection_name=self.collection_name,
                vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter=qdrant_filter,
                with_payload=True,
            )

        results: List[IsoSearchResultItem] = []
        for hit in search_results:
            payload = hit.payload or {}
            metadata_dict = payload.get("metadata", {})
            metadata_obj = IsoFactMetadata(**metadata_dict)

            results.append(
                IsoSearchResultItem(
                    fact_id=payload.get("fact_id", str(hit.id)),
                    score=round(float(hit.score), 4),
                    fact_summary=payload.get("fact_summary", ""),
                    fact_content=payload.get("fact_content", ""),
                    formatted_context=payload.get("formatted_context", ""),
                    metadata=metadata_obj
                )
            )

        return results

    def get_collection_stats(self) -> CollectionStatsResponse:
        """Get collection metrics and counts."""
        self.ensure_collection()
        info = self.client.get_collection(collection_name=self.collection_name)
        points_count = getattr(info, "points_count", 0) or 0
        vectors_count = getattr(info, "indexed_vectors_count", getattr(info, "vectors_count", points_count)) or points_count
        status_val = info.status.name if hasattr(info.status, "name") else str(info.status)

        return CollectionStatsResponse(
            collection_name=self.collection_name,
            vectors_count=vectors_count,
            points_count=points_count,
            status=status_val,
            vector_dimension=settings.EMBEDDING_VECTOR_SIZE,
        )

    def reset_collection(self) -> bool:
        """Re-create the collection from scratch."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except UnexpectedResponse:
            pass
        self.ensure_collection()
        return True


qdrant_service = QdrantService()
