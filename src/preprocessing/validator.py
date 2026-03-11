"""
Data Validator
Validates all chunks have required metadata fields, no empty descriptions,
and no duplicate chunk IDs. Run before every vector store rebuild.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHUNKS_FILE = Path("data/chunks/all_chunks.json")

REQUIRED_FIELDS = [
    "chunk_id",
    "text",
    "species_name",
    "scientific_name",
    "section_type",
    "taxonomic_group",
    "geographic_regions",
    "conservation_status",
    "source_urls",
]


def validate_chunks(chunks: list[dict]) -> dict:
    """Validate all chunks and return a report."""
    errors = []
    warnings = []
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in chunk:
                errors.append(f"Chunk {i}: missing required field '{field}'")
            elif chunk[field] is None:
                errors.append(f"Chunk {i}: field '{field}' is None")

        # Check text is not empty
        text = chunk.get("text", "")
        if not text or len(text.strip()) < 30:
            errors.append(
                f"Chunk {i} ({chunk.get('species_name', '?')}): text too short ({len(text)} chars)"
            )

        # Check chunk_id uniqueness
        cid = chunk.get("chunk_id", "")
        if cid in chunk_ids:
            errors.append(f"Chunk {i}: duplicate chunk_id '{cid}'")
        chunk_ids.append(cid)

        # Check species_name is not empty
        if not chunk.get("species_name", "").strip():
            errors.append(f"Chunk {i}: empty species_name")

        # Warnings for optional but recommended fields
        if not chunk.get("geographic_regions"):
            warnings.append(f"Chunk {i} ({chunk.get('species_name', '?')}): no geographic_regions")

        if not chunk.get("conservation_status"):
            warnings.append(f"Chunk {i} ({chunk.get('species_name', '?')}): no conservation_status")

        # Check token estimate is reasonable
        tokens = chunk.get("token_estimate", 0)
        if tokens > 800:
            warnings.append(
                f"Chunk {i} ({chunk.get('species_name', '?')}): very large ({tokens} tokens)"
            )

    # Summary stats
    groups = Counter(c.get("taxonomic_group", "unknown") for c in chunks)
    sections = Counter(c.get("section_type", "unknown") for c in chunks)
    species_count = len(set(c.get("scientific_name", "") for c in chunks))

    report = {
        "total_chunks": len(chunks),
        "unique_species": species_count,
        "errors": len(errors),
        "warnings": len(warnings),
        "error_details": errors[:50],  # Cap at 50 for readability
        "warning_details": warnings[:50],
        "chunks_by_group": dict(groups),
        "chunks_by_section": dict(sections),
        "validation_passed": len(errors) == 0,
    }

    return report


def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.error("Run the chunker first: python -m src.preprocessing.chunker")
        sys.exit(1)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Validating {len(chunks)} chunks...")
    report = validate_chunks(chunks)

    # Save report
    report_file = Path("data/chunks/validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print results
    if report["validation_passed"]:
        logger.info("VALIDATION PASSED")
        logger.info(f"  {report['total_chunks']} chunks, {report['unique_species']} species")
        logger.info(f"  Groups: {report['chunks_by_group']}")
        if report["warnings"]:
            logger.warning(f"  {report['warnings']} warnings (non-blocking)")
    else:
        logger.error("VALIDATION FAILED")
        logger.error(f"  {report['errors']} errors found:")
        for err in report["error_details"][:10]:
            logger.error(f"    - {err}")
        sys.exit(1)

    if report["warnings"] > 0:
        logger.warning("Sample warnings:")
        for warn in report["warning_details"][:5]:
            logger.warning(f"  - {warn}")


if __name__ == "__main__":
    main()
