from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IsoFactMetadata(BaseModel):
    """Metadata schema representing structural headings and context for an ISO fact."""
    iso_standard: str = Field(
        ...,
        description="ISO standard identifier (e.g., 'ISO 9001:2015', 'ISO 27001:2022')",
        example="ISO 9001:2015"
    )
    clause_number: str = Field(
        ...,
        description="Clause or sub-clause number (e.g., '7.1.5.2', '8.5.1')",
        example="7.1.5.2"
    )
    clause_title: str = Field(
        ...,
        description="Title or heading of the clause",
        example="Measurement Traceability"
    )
    sub_clause: Optional[str] = Field(
        None,
        description="Sub-clause details or item indicator if applicable",
        example="7.1.5.2 (a) Calibration standards"
    )
    topic: str = Field(
        ...,
        description="Functional topic or domain category",
        example="Calibration & Quality Control"
    )
    odoo_module: Optional[str] = Field(
        None,
        description="Associated Odoo technical module name (e.g., 'quality_control', 'mrp', 'stock')",
        example="quality_control"
    )
    odoo_model: Optional[str] = Field(
        None,
        description="Associated Odoo model name (e.g., 'quality.check', 'maintenance.equipment')",
        example="quality.check"
    )
    odoo_process: Optional[str] = Field(
        None,
        description="Specific workflow or process in Odoo that relates to this ISO requirement",
        example="Periodic calibration validation for test instruments"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Keywords or tags for fast categorization",
        example=["calibration", "traceability", "equipment", "audit-mandatory"]
    )
    extra_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom key-value pairs for flexible metadata extensions",
        example={"mandatory_evidence": "Calibration certificate record", "audit_cycle": "Annual"}
    )


class IsoFactCreate(BaseModel):
    """Schema for creating a single atomic ISO fact with metadata."""
    fact_id: Optional[str] = Field(
        None,
        description="Optional unique identifier (UUID or reference string). Auto-generated if omitted.",
        example="iso9001-7-1-5-2-fact-01"
    )
    fact_summary: str = Field(
        ...,
        description="Concise 1-sentence title or summary of this standalone fact",
        example="Measuring equipment must be calibrated against certified international or national measurement standards."
    )
    fact_content: str = Field(
        ...,
        description="The detailed independent atomic fact text and compliance guideline",
        example="When measurement traceability is a requirement, measuring equipment shall be calibrated or verified, or both, at specified intervals, or prior to use, against measurement standards traceable to international or national standards. In Odoo, this is enforced through maintenance equipment calibration schedules and quality check passes."
    )
    metadata: IsoFactMetadata


class IsoFactBulkCreate(BaseModel):
    """Schema for bulk ingestion of ISO facts."""
    documents: List[IsoFactCreate]


class IngestionResultItem(BaseModel):
    fact_id: str
    status: str
    iso_standard: str
    clause_number: str


class IngestionResponse(BaseModel):
    total_received: int
    total_indexed: int
    results: List[IngestionResultItem]


class IsoSearchFilter(BaseModel):
    """Metadata filter criteria for semantic search queries."""
    iso_standard: Optional[str] = Field(None, example="ISO 9001:2015")
    clause_number: Optional[str] = Field(None, example="7.1.5.2")
    topic: Optional[str] = Field(None, example="Calibration & Quality Control")
    odoo_module: Optional[str] = Field(None, example="quality_control")
    odoo_model: Optional[str] = Field(None, example="quality.check")
    tag: Optional[str] = Field(None, example="calibration")


class IsoSearchRequest(BaseModel):
    """Semantic search query request."""
    query: str = Field(
        ...,
        description="Natural language query or requirement query",
        example="How does Odoo handle measuring equipment calibration for ISO audit?"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    score_threshold: Optional[float] = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score threshold"
    )
    filters: Optional[IsoSearchFilter] = None


class IsoSearchResultItem(BaseModel):
    fact_id: str
    score: float
    fact_summary: str
    fact_content: str
    formatted_context: str
    metadata: IsoFactMetadata


class IsoSearchResponse(BaseModel):
    query: str
    total_found: int
    results: List[IsoSearchResultItem]


class CollectionStatsResponse(BaseModel):
    collection_name: str
    vectors_count: int
    points_count: int
    status: str
    vector_dimension: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    qdrant_status: str
