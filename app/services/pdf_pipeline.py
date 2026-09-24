import logging
import sys
import time
from typing import Any, Dict, List, Optional

from app.models.schemas import IngestionResultItem, IsoFactCreate
from app.services.embedding_service import embedding_service
from app.services.llm_extractor import llm_extractor
from app.services.pdf_extractor import pdf_parser
from app.services.qdrant_service import qdrant_service

# Setup logger with clean stdout formatting
logger = logging.getLogger("iso_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PDFIngestionPipeline:
    """
    End-to-end pipeline orchestrating PDF layout parsing, LLM-based 
    fact extraction with live progress tracking, FastEmbed vectorization, and Qdrant upsertion.
    """

    @staticmethod
    def _draw_progress_bar(current: int, total: int, bar_length: int = 25) -> str:
        """Renders a clean ASCII progress bar."""
        if total == 0:
            return "[-------------------------] 0%"
        fraction = current / total
        filled = int(fraction * bar_length)
        bar = "=" * filled + (">" if filled < bar_length else "") + " " * (bar_length - filled - 1 if filled < bar_length else 0)
        percent = int(fraction * 100)
        return f"[{bar}] {percent:3d}%"

    def parse_pdf_to_facts(
        self,
        pdf_bytes: bytes,
        default_iso_standard: str = "ISO 9001:2015",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[IsoFactCreate], int]:
        """
        Extracts atomic ISO facts from PDF bytes with rich real-time progress logging.
        """
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("🚀 [PDF INGESTION] Starting PDF Parsing & LLM Fact Extraction...")

        pages = pdf_parser.parse_pdf_bytes(pdf_bytes)
        if not pages:
            logger.warning("⚠️  [PDF INGESTION] PDF contained no readable or valid text pages.")
            return [], 0

        logger.info(f"📄 [PDF Parser] Extracted {len(pages)} readable pages from document.")

        windows = pdf_parser.build_windows(pages)
        total_windows = len(windows)
        logger.info(f"🧩 [Window Builder] Grouped pages into {total_windows} processing windows for LLM.")
        logger.info("-" * 70)

        all_facts: List[IsoFactCreate] = []
        start_llm_time = time.time()

        for idx, win in enumerate(windows, start=1):
            w_start = time.time()
            page_tags = ", ".join(f"P{p}" for p in win["pages"])
            char_count = len(win["text"])

            pbar = self._draw_progress_bar(idx - 1, total_windows)
            logger.info(f"{pbar} Window {idx}/{total_windows} | Pages: [{page_tags}] ({char_count} chars) -> Querying Gemma3:12b...")

            facts = llm_extractor.extract_facts_from_window(
                text=win["text"],
                source_pages=win["pages"],
                default_iso_standard=default_iso_standard,
                document_metadata=document_metadata,
            )
            all_facts.extend(facts)

            w_elapsed = round(time.time() - w_start, 2)
            clauses_found = [f.metadata.clause_number for f in facts if f.metadata.clause_number]
            clauses_str = ", ".join(clauses_found[:4]) if clauses_found else "None"
            if len(clauses_found) > 4:
                clauses_str += f" (+{len(clauses_found) - 4} more)"

            # Estimate Remaining Time
            avg_time = (time.time() - start_llm_time) / idx
            remaining_secs = int(avg_time * (total_windows - idx))

            pbar_done = self._draw_progress_bar(idx, total_windows)
            logger.info(
                f"{pbar_done} Window {idx}/{total_windows} Done in {w_elapsed}s | "
                f"+{len(facts)} facts (Clauses: {clauses_str}) | Total Facts: {len(all_facts)} | ETA: ~{remaining_secs}s"
            )

        total_elapsed = round(time.time() - t0, 2)
        logger.info("-" * 70)
        logger.info(f"✅ [Extraction Complete] Extracted {len(all_facts)} atomic facts across {len(pages)} pages in {total_elapsed}s.")
        return all_facts, len(pages)

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        default_iso_standard: str = "ISO 9001:2015",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end PDF ingestion with visual logging:
        1. Layout-aware parse & OCR
        2. LLM Gemma3:12b fact extraction
        3. FastEmbed vectorization
        4. Qdrant indexing
        """
        t_start = time.time()
        meta = {
            **(document_metadata or {}),
            "filename": filename,
        }

        logger.info(f"📁 Processing File: '{filename}' | Standard: {default_iso_standard}")

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

        # Step 3: FastEmbed Vectorization
        logger.info(f"🧠 [FastEmbed] Vectorizing {len(facts)} facts with contextual ISO headers...")
        t_embed = time.time()
        vectors, formatted_texts = embedding_service.embed_iso_documents(facts)
        embed_time = round(time.time() - t_embed, 2)
        logger.info(f"⚡ [FastEmbed] Embedding generated in {embed_time}s.")

        # Step 4: Qdrant Indexing
        logger.info(f"💾 [Qdrant] Upserting {len(facts)} points with payload indexes into collection...")
        t_qdrant = time.time()
        upsert_results = qdrant_service.upsert_iso_facts(
            documents=facts,
            vectors=vectors,
            formatted_texts=formatted_texts,
        )
        qdrant_time = round(time.time() - t_qdrant, 2)
        total_time = round(time.time() - t_start, 2)

        logger.info(f"🎯 [Qdrant] Indexed {len(upsert_results)} points in {qdrant_time}s.")
        logger.info(f"🎉 [Pipeline Finished] Successfully indexed '{filename}' ({len(upsert_results)} facts) in {total_time}s total!")
        logger.info("=" * 70)

        return {
            "status": "success",
            "filename": filename,
            "pages_parsed": pages_count,
            "total_facts_extracted": len(facts),
            "total_indexed": len(upsert_results),
            "results": [IngestionResultItem(**r) for r in upsert_results],
        }


pdf_pipeline = PDFIngestionPipeline()

