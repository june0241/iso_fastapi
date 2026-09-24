from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.models.schemas import (
    CollectionStatsResponse,
    IngestionResponse,
    IngestionResultItem,
    IsoFactBulkCreate,
    IsoFactCreate,
    IsoSearchRequest,
    IsoSearchResponse,
    PdfIngestionResponse,
    PdfPreviewResponse,
)
from app.services.embedding_service import embedding_service
from app.services.pdf_pipeline import pdf_pipeline
from app.services.qdrant_service import qdrant_service

router = APIRouter(prefix="/iso", tags=["ISO Reference Embeddings"])


@router.post(
    "/upload/pdf",
    response_model=PdfIngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload ISO Reference PDF, extract atomic facts via Gemma3:12b, and index in Qdrant"
)
async def upload_and_ingest_iso_pdf(
    file: UploadFile = File(..., description="ISO Reference or SOP PDF document"),
    default_iso_standard: str = Form("ISO 9001:2015", description="Default ISO Standard identifier"),
    department_or_owner: Optional[str] = Form(None, description="Optional department or auditor name"),
):
    """
    Complete layout-resilient PDF ingestion pipeline:
    1. PyMuPDF visual order extraction + Tesseract OCR fallback for scanned pages.
    2. Context windowing with noise stripping.
    3. Ollama Gemma3:12b extraction into independent facts with Odoo mappings.
    4. FastEmbed vectorization and upsertion to Qdrant.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    try:
        pdf_bytes = await file.read()
        doc_metadata = {}
        if department_or_owner:
            doc_metadata["department_or_owner"] = department_or_owner

        result = pdf_pipeline.ingest_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            default_iso_standard=default_iso_standard,
            document_metadata=doc_metadata,
        )
        return PdfIngestionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and ingest PDF: {str(e)}"
        )


@router.post(
    "/parse/pdf-preview",
    response_model=PdfPreviewResponse,
    summary="Dry-run: Parse ISO PDF & extract atomic facts with Gemma3:12b without storing in Qdrant"
)
async def preview_iso_pdf_facts(
    file: UploadFile = File(..., description="ISO Reference PDF to preview"),
    default_iso_standard: str = Form("ISO 9001:2015", description="Default ISO Standard identifier"),
):
    """
    Extracts atomic ISO facts from PDF without saving to Qdrant. 
    Useful for inspecting or validating extracted metadata headings in Swagger UI.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    try:
        pdf_bytes = await file.read()
        facts, pages_count = pdf_pipeline.parse_pdf_to_facts(
            pdf_bytes=pdf_bytes,
            default_iso_standard=default_iso_standard,
        )
        return PdfPreviewResponse(
            status="success",
            filename=file.filename,
            pages_parsed=pages_count,
            total_facts_extracted=len(facts),
            facts=facts,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview PDF facts: {str(e)}"
        )


@router.post(
    "/embed",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Embed and index a single ISO reference fact"
)
async def embed_single_iso_fact(doc: IsoFactCreate):
    """
    Ingest a single atomic ISO fact document with metadata headings.
    The document is vectorized and indexed into Qdrant.
    """
    try:
        vectors, formatted_texts = embedding_service.embed_iso_documents([doc])
        results = qdrant_service.upsert_iso_facts(
            documents=[doc],
            vectors=vectors,
            formatted_texts=formatted_texts,
        )
        return IngestionResponse(
            total_received=1,
            total_indexed=len(results),
            results=[IngestionResultItem(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed and index ISO fact: {str(e)}"
        )


@router.post(
    "/embed/bulk",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Embed and index a batch of ISO reference facts"
)
async def embed_bulk_iso_facts(payload: IsoFactBulkCreate):
    """
    Ingest a batch of atomic ISO reference facts with metadata headings.
    Batch embeds the texts and upserts all points to Qdrant.
    """
    if not payload.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document list cannot be empty."
        )

    try:
        vectors, formatted_texts = embedding_service.embed_iso_documents(payload.documents)
        results = qdrant_service.upsert_iso_facts(
            documents=payload.documents,
            vectors=vectors,
            formatted_texts=formatted_texts,
        )
        return IngestionResponse(
            total_received=len(payload.documents),
            total_indexed=len(results),
            results=[IngestionResultItem(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch embed and index ISO facts: {str(e)}"
        )


@router.post(
    "/search",
    response_model=IsoSearchResponse,
    summary="Semantic search over ISO facts with metadata filters"
)
async def search_iso_facts(search_req: IsoSearchRequest):
    """
    Perform semantic vector search on ISO reference facts.
    Supports filtering by ISO standard, clause number, topic, Odoo module, and tags.
    """
    try:
        query_vector = embedding_service.embed_query(search_req.query)
        hits = qdrant_service.search_iso_facts(
            query_vector=query_vector,
            top_k=search_req.top_k,
            score_threshold=search_req.score_threshold,
            filters=search_req.filters,
        )
        return IsoSearchResponse(
            query=search_req.query,
            total_found=len(hits),
            results=hits,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute semantic search: {str(e)}"
        )


@router.get(
    "/collection/stats",
    response_model=CollectionStatsResponse,
    summary="Get Qdrant ISO collection statistics"
)
async def get_collection_statistics():
    """Retrieve collection status, vectors count, and point counts from Qdrant."""
    try:
        return qdrant_service.get_collection_stats()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch collection stats: {str(e)}"
        )


@router.delete(
    "/collection",
    summary="Reset and clear the ISO Qdrant collection"
)
async def reset_collection():
    """Delete and recreate the ISO collection in Qdrant (Caution: destructive)."""
    try:
        qdrant_service.reset_collection()
        return {"status": "success", "message": "Collection cleared and reinitialized"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset collection: {str(e)}"
        )
