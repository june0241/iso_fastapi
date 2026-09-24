import json
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import ollama
except ImportError:
    ollama = None

from app.core.config import settings
from app.models.schemas import IsoFactCreate, IsoFactMetadata

logger = logging.getLogger("llm_extractor")


ISO_EXTRACTION_PROMPT = """\
You are an expert ISO compliance auditor and Odoo ERP architect.
Extract all concrete, independent compliance requirements, clauses, and Odoo ERP mappings from the provided text into a JSON array.

Each item in the JSON array MUST follow this exact schema:
{{
  "iso_standard": "{default_iso_standard}",
  "clause_number": "<clause number, e.g. 4.1, 7.1.5.2, 8.5.1, A.9.2.1>",
  "clause_title": "<heading / title of the clause, max 100 chars>",
  "sub_clause": "<sub-clause identifier/title or null>",
  "topic": "<functional category, e.g. Quality Control, Calibration, Access Control, Document Control, Traceability>",
  "odoo_module": "<matching Odoo technical module name, e.g. quality_control, maintenance, stock, mrp, hr, base, or general>",
  "odoo_model": "<technical Odoo model if applicable, e.g. quality.check, maintenance.equipment, stock.lot, res.users, or null>",
  "odoo_process": "<brief description of the workflow or mechanism in Odoo that enforces this requirement>",
  "fact_summary": "<concise 1-sentence standalone summary of the fact, max 120 chars>",
  "fact_content": "<detailed, self-contained independent fact describing the requirement and compliance guideline>",
  "tags": ["<keyword1>", "<keyword2>"]
}}

Rules:
1. Extract distinct requirements individually as atomic, standalone facts.
2. Ensure every fact is fully self-explanatory without needing the original PDF.
3. If an explicit ISO standard (e.g. ISO 9001:2015, ISO 27001:2022) is detected in text, use that for 'iso_standard'; otherwise default to "{default_iso_standard}".
4. Output ONLY the JSON array. Do not include markdown codeblocks, notes, or explanations.

Document Text:
{text}
"""


class LLMFactExtractor:
    """
    LLM-powered extractor using Ollama (Gemma3:12b) to transform raw document 
    windows into structured IsoFactCreate objects with JSON repair and consensus.
    """

    def __init__(self):
        self.ollama_client = None
        if ollama is not None:
            self.ollama_client = ollama.Client(host=settings.OLLAMA_HOST)

    @staticmethod
    def clean_json_markdown(raw: str) -> str:
        """Strips markdown code blocks and trims whitespace."""
        raw = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            return m.group(1).strip()
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            return m.group(0).strip()
        return raw

    @staticmethod
    def repair_truncated_json(raw: str) -> str:
        """Repairs incomplete JSON arrays and balances brackets."""
        raw = raw.rstrip().rstrip(",").rstrip()
        if raw.count("{") > raw.count("}"):
            last_complete = max(raw.rfind("},"), raw.rfind("}"))
            if last_complete != -1:
                raw = raw[: last_complete + 1]
        open_brackets = raw.count("[") - raw.count("]")
        raw += "]" * max(open_brackets, 0)
        return raw

    def call_llm(
        self,
        text: str,
        default_iso_standard: str = "ISO 9001:2015",
        temperature: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Invokes Ollama Gemma3:12b to parse document text into structured JSON."""
        if self.ollama_client is None:
            raise RuntimeError(
                "Ollama library is not installed or initialized. Please run `pip install ollama`."
            )

        prompt = ISO_EXTRACTION_PROMPT.format(
            text=text,
            default_iso_standard=default_iso_standard,
        )

        response = self.ollama_client.chat(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "num_predict": 4096,
                "num_ctx": 8192,
            },
        )

        content = response["message"]["content"]
        cleaned = self.clean_json_markdown(content)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = self.repair_truncated_json(cleaned)
            parsed = json.loads(repaired)

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array from LLM, received {type(parsed)}")

        return parsed

    def extract_facts_from_window(
        self,
        text: str,
        source_pages: List[int],
        default_iso_standard: str = "ISO 9001:2015",
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[IsoFactCreate]:
        """
        Extracts ISO facts from a window text, validates through Pydantic, and returns IsoFactCreate list.
        """
        all_raw_facts: List[Dict[str, Any]] = []

        for p in range(settings.EXTRACTION_PASSES):
            temp = settings.LLM_TEMPERATURE + (p * 0.05)
            try:
                facts = self.call_llm(
                    text=text,
                    default_iso_standard=default_iso_standard,
                    temperature=temp,
                )
                all_raw_facts.extend(facts)
            except Exception as exc:
                logger.warning("LLM extraction pass %d failed: %s", p + 1, exc)

        if not all_raw_facts:
            return []

        validated_facts: List[IsoFactCreate] = []
        doc_meta = document_metadata or {}
        first_page = source_pages[0] if source_pages else 1

        for raw in all_raw_facts:
            try:
                clause_num = str(raw.get("clause_number") or "General").strip()
                fact_summary = str(raw.get("fact_summary") or "").strip()
                fact_content = str(raw.get("fact_content") or "").strip()

                if not fact_summary and not fact_content:
                    continue

                if not fact_summary:
                    fact_summary = fact_content[:100] + "..."
                if not fact_content:
                    fact_content = fact_summary

                metadata = IsoFactMetadata(
                    iso_standard=str(raw.get("iso_standard") or default_iso_standard).strip(),
                    clause_number=clause_num,
                    clause_title=str(raw.get("clause_title") or f"Clause {clause_num}").strip(),
                    sub_clause=raw.get("sub_clause"),
                    topic=str(raw.get("topic") or "General Compliance").strip(),
                    odoo_module=raw.get("odoo_module") or "general",
                    odoo_model=raw.get("odoo_model"),
                    odoo_process=raw.get("odoo_process"),
                    tags=raw.get("tags") if isinstance(raw.get("tags"), list) else [],
                    extra_attributes={
                        **doc_meta,
                        "source_page": first_page,
                        "source_pages": source_pages,
                    }
                )

                fact_id = f"fact-{uuid4().hex[:12]}"

                validated_facts.append(
                    IsoFactCreate(
                        fact_id=fact_id,
                        fact_summary=fact_summary,
                        fact_content=fact_content,
                        metadata=metadata
                    )
                )
            except Exception as e:
                logger.warning("Failed to validate extracted fact item: %s", e)

        return validated_facts


llm_extractor = LLMFactExtractor()
