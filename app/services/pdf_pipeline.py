import logging
from typing import Any, Dict, List, Optional

from app.models.schemas import IngestionResultItem, IsoFactCreate
from app.services.embedding_service import embedding_service
from app.services.llm_extractor import llm_extractor
from app.services.pdf_extractor import pdf_parser
from app.services.qdrant_service import qdrant_service

logger = logging.getLogger("pdf_pipeline")


class PDFIngestionPipeline:
    """
    End-to-end pipeline orchestrating PDF layout parsing, LLM-based 
    fact extraction, FastEmbed vectorization, and Qdrant upsertion.
    """

    def parse_pdf_to_facts(
        self,
        pdf_bytes: bytes,
        default_iso_standard: str = "ISO 9001:2015",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[IsoFactCreate], int]:
        """
        Extracts atomic ISO facts from PDF bytes without upserting to Qdrant.
        Returns (facts_list, total_pages_parsed).
        """
        pages = pdf_parser.parse_pdf_bytes(pdf_bytes)
        if not pages:
            return [], 0

        windows = pdf_parser.build_windows(pages)
        all_facts: List[IsoFactCreate] = []

        for win in windows:
            facts = llm_extractor.extract_facts_from_window(
                text=win["text"],
                source_pages=win["pages"],
                default_iso_standard=default_iso_standard,
                document_metadata=document_metadata,
            )
            all_facts.extend(facts)

        return all_facts, len(pages)

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        default_iso_standard: str = "ISO 9001:2015",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end PDF ingestion:
        1. Layout-aware parse & OCR
        2. LLM Gemma3:12b fact extraction
        3. FastEmbed vectorization
        4. Qdrant indexing
        """
        meta = {
            **(document_metadata or {}),
            "filename": filename,
        }

        facts, pages_count = self.parse_pdf_to_facts(
            pdf_bytes=pdf_bytes,
            default_iso_standard=default_iso_standard,
            document_metadata=meta,
        )

        if not facts:
            return {
                "status": "empty",
                "filename": filename,
                "pages_parsed": pages_count,
                "total_facts_extracted": 0,
                "total_indexed": 0,
                "results": [],
            }

        vectors, formatted_texts = embedding_service.embed_iso_documents(facts)
        upsert_results = qdrant_service.upsert_iso_facts(
            documents=facts,
            vectors=vectors,
            formatted_texts=formatted_texts,
        )

        return {
            "status": "success",
            "filename": filename,
            "pages_parsed": pages_count,
            "total_facts_extracted": len(facts),
            "total_indexed": len(upsert_results),
            "results": [IngestionResultItem(**r) for r in upsert_results],
        }


pdf_pipeline = PDFIngestionPipeline()
