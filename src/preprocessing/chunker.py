"""
Species-Aware Chunker
Splits processed species data into chunks suitable for RAG.
CRITICAL: Never let a chunk cross species boundaries.
Each chunk carries full metadata for filtering and observability.
"""

import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/chunks")

# Chunking parameters
TARGET_TOKENS = 500  # Target chunk size in approximate tokens
MAX_TOKENS = 600  # Maximum chunk size
OVERLAP_TOKENS = 80  # Overlap between chunks of same species/section
APPROX_CHARS_PER_TOKEN = 4  # Rough estimate for English text


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if not text or len(text) < 50:
        return [text] if text else []

    if estimate_tokens(text) <= MAX_TOKENS and len(text) <= max_chars:
        return [text]

    # Split by sentences
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current) > 20:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    if not sentences:
        return [text]

    # Group sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

        if len(test_chunk) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: include last part of previous chunk
            overlap_text = (
                current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else ""
            )
            current_chunk = f"{overlap_text} {sentence}".strip()
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def generate_chunk_id(species_name: str, section: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    content = f"{species_name}|{section}|{index}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def chunk_species(species: dict) -> list[dict]:
    """Create chunks from a single species record. Never crosses species boundaries."""
    species_name = species.get("primary_name", "Unknown")
    scientific_name = species.get("scientific_name", "")
    conservation = species.get("conservation_status", "")
    regions = species.get("geographic_regions", [])
    sources = species.get("sources", {})

    # Base metadata for all chunks from this species
    base_metadata = {
        "species_name": species_name,
        "scientific_name": scientific_name,
        "taxonomic_group": species.get("taxonomic_group", ""),
        "conservation_status": conservation,
        "geographic_regions": regions,
        "family": species.get("taxonomy", {}).get("family", ""),
        "order": species.get("taxonomy", {}).get("order", ""),
    }

    chunks = []
    description = species.get("description", {})

    # Section mapping: section_key -> human-readable section type
    section_types = {
        "introduction": "Overview",
        "physical_description": "Physical Description",
        "habitat": "Habitat & Distribution",
        "behavior": "Behavior & Ecology",
        "diet": "Diet & Feeding",
        "reproduction": "Reproduction & Breeding",
        "conservation": "Conservation Status",
    }

    max_chars = TARGET_TOKENS * APPROX_CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * APPROX_CHARS_PER_TOKEN

    for section_key, section_label in section_types.items():
        text = description.get(section_key, "")
        if not text or len(text.strip()) < 30:
            continue

        # Prepend structured header for better embedding context
        header = f"Species: {species_name} ({scientific_name}) | Section: {section_label}"
        if regions:
            header += f" | Regions: {', '.join(regions[:5])}"
        if conservation:
            header += f" | Status: {conservation}"

        text_chunks = split_text(text, max_chars, overlap_chars)

        for i, chunk_text in enumerate(text_chunks):
            full_text = f"{header}\n\n{chunk_text}"
            chunk_id = generate_chunk_id(scientific_name or species_name, section_key, i)

            chunk = {
                "chunk_id": chunk_id,
                "text": full_text,
                "raw_text": chunk_text,
                "section_type": section_label,
                "chunk_index": i,
                "total_chunks_in_section": len(text_chunks),
                "token_estimate": estimate_tokens(full_text),
                "source_urls": [url for url in sources.values() if url],
                **base_metadata,
            }
            chunks.append(chunk)

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed species data
    all_species_file = INPUT_DIR / "all_species.json"
    if not all_species_file.exists():
        logger.error("No processed data found. Run cleaner.py first.")
        return

    with open(all_species_file, encoding="utf-8") as f:
        all_species = json.load(f)

    logger.info(f"Chunking {len(all_species)} species...")

    all_chunks = []
    species_with_chunks = 0
    empty_species = []

    for species in all_species:
        chunks = chunk_species(species)
        if chunks:
            all_chunks.extend(chunks)
            species_with_chunks += 1
        else:
            empty_species.append(species.get("primary_name", "Unknown"))

    # Save all chunks
    with open(OUTPUT_DIR / "all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Stats
    token_counts = [c["token_estimate"] for c in all_chunks]
    stats = {
        "total_chunks": len(all_chunks),
        "species_with_chunks": species_with_chunks,
        "species_without_chunks": len(empty_species),
        "chunks_by_section": {},
        "avg_tokens_per_chunk": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "chunks_by_group": {},
    }

    for chunk in all_chunks:
        sec = chunk["section_type"]
        stats["chunks_by_section"][sec] = stats["chunks_by_section"].get(sec, 0) + 1
        grp = chunk["taxonomic_group"]
        stats["chunks_by_group"][grp] = stats["chunks_by_group"].get(grp, 0) + 1

    with open(OUTPUT_DIR / "chunking_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if empty_species:
        with open(OUTPUT_DIR / "species_without_chunks.json", "w", encoding="utf-8") as f:
            json.dump(empty_species, f, indent=2)

    logger.info(f"Chunking complete: {len(all_chunks)} chunks from {species_with_chunks} species")
    logger.info(f"Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.0f}")
    logger.info(f"By section: {json.dumps(stats['chunks_by_section'], indent=2)}")


if __name__ == "__main__":
    main()
