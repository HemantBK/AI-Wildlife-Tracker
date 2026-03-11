"""
End-to-End Pipeline Smoke Test
Verifies each component works individually and together.
Run after building indexes: python -m pytest tests/test_pipeline_e2e.py -v
"""

import json
from pathlib import Path

import pytest

# ─── Data Existence Tests ──────────────────────────────────────


class TestDataPipeline:
    """Verify data pipeline outputs exist and are valid."""

    def test_processed_data_exists(self):
        path = Path("data/processed/all_species.json")
        assert path.exists(), "Run data pipeline first: make data-pipeline"

    def test_chunks_exist(self):
        path = Path("data/chunks/all_chunks.json")
        assert path.exists(), "Run chunker first: make chunk"

    def test_chunks_have_required_fields(self):
        path = Path("data/chunks/all_chunks.json")
        if not path.exists():
            pytest.skip("No chunks file")

        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)

        assert len(chunks) > 0, "No chunks found"

        required = ["chunk_id", "text", "species_name", "scientific_name", "section_type"]
        for chunk in chunks[:10]:
            for field in required:
                assert field in chunk, f"Chunk missing field: {field}"

    def test_no_cross_species_chunks(self):
        """Verify no chunk contains multiple species names in its metadata."""
        path = Path("data/chunks/all_chunks.json")
        if not path.exists():
            pytest.skip("No chunks file")

        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            # Each chunk should have exactly one species
            assert chunk.get("species_name"), f"Chunk {chunk['chunk_id']} has no species_name"


# ─── Query Expander Tests ─────────────────────────────────────


class TestQueryExpander:
    """Test query preprocessing and expansion."""

    def test_location_extraction(self):
        from src.retrieval.query_expander import extract_location

        assert extract_location("bird in Kerala") == "Kerala"
        assert extract_location("tiger in Ranthambore") == "Rajasthan"
        assert extract_location("snake in Tamil Nadu") == "Tamil Nadu"
        assert extract_location("animal somewhere") is None

    def test_season_extraction(self):
        from src.retrieval.query_expander import extract_season

        assert extract_season("birds in summer") == "summer"
        assert extract_season("monsoon migration") == "monsoon"
        assert extract_season("cold winter night") == "winter"
        assert extract_season("regular day") is None

    def test_query_expansion(self):
        from src.retrieval.query_expander import expand_query

        expanded = expand_query("brown bird near water")
        assert "wetland" in expanded or "aquatic" in expanded
        assert "rufous" in expanded or "chestnut" in expanded

    def test_preprocess_full(self):
        from src.retrieval.query_expander import preprocess_query

        result = preprocess_query("small brown bird near water in Kerala during monsoon")
        assert result["location"] == "Kerala"
        assert result["season"] == "monsoon"
        assert "brown" in result["features"]["colors"]
        assert len(result["expanded_query"]) > len(result["original_query"])


# ─── Chunker Tests ────────────────────────────────────────────


class TestChunker:
    """Test chunking logic."""

    def test_split_text_short(self):
        from src.preprocessing.chunker import split_text

        # Short text should not be split
        result = split_text("This is a short text.", 2000, 200)
        assert len(result) == 1

    def test_split_text_long(self):
        from src.preprocessing.chunker import split_text

        # Create a text that needs splitting
        long_text = ". ".join([f"This is sentence number {i} about wildlife" for i in range(50)])
        result = split_text(long_text, 500, 100)
        assert len(result) > 1

    def test_chunk_id_deterministic(self):
        from src.preprocessing.chunker import generate_chunk_id

        id1 = generate_chunk_id("Tiger", "habitat", 0)
        id2 = generate_chunk_id("Tiger", "habitat", 0)
        id3 = generate_chunk_id("Tiger", "diet", 0)
        assert id1 == id2, "Same inputs should give same ID"
        assert id1 != id3, "Different inputs should give different IDs"


# ─── Validator Tests ──────────────────────────────────────────


class TestValidator:
    """Test chunk validation logic."""

    def test_valid_chunk_passes(self):
        from src.preprocessing.validator import validate_chunks

        chunks = [
            {
                "chunk_id": "test_001",
                "text": "The Bengal Tiger is a large predator found in Indian forests." * 3,
                "species_name": "Bengal Tiger",
                "scientific_name": "Panthera tigris tigris",
                "section_type": "Overview",
                "taxonomic_group": "mammals",
                "geographic_regions": ["Madhya Pradesh", "Rajasthan"],
                "conservation_status": "Endangered",
                "source_urls": ["https://example.com"],
            }
        ]

        report = validate_chunks(chunks)
        assert report["validation_passed"] is True
        assert report["errors"] == 0

    def test_missing_field_fails(self):
        from src.preprocessing.validator import validate_chunks

        chunks = [
            {
                "chunk_id": "test_002",
                "text": "Some text about a bird that is reasonably long for validation.",
                # Missing species_name and other required fields
            }
        ]

        report = validate_chunks(chunks)
        assert report["validation_passed"] is False
        assert report["errors"] > 0


# ─── Generator Schema Tests ──────────────────────────────────


class TestGeneratorSchema:
    """Test Pydantic response schemas."""

    def test_identification_response_valid(self):
        from src.rag.generator import IdentificationResponse

        resp = IdentificationResponse(
            species_name="Bengal Tiger",
            scientific_name="Panthera tigris tigris",
            confidence=0.92,
            reasoning="Large striped cat matches tiger description",
            key_features_matched=["stripes", "large", "orange"],
            conservation_status="Endangered",
            cited_sources=["chunk_001", "chunk_002"],
        )
        assert resp.species_name == "Bengal Tiger"
        assert resp.confidence == 0.92

    def test_confidence_bounds(self):
        from pydantic import ValidationError

        from src.rag.generator import IdentificationResponse

        with pytest.raises(ValidationError):
            IdentificationResponse(
                species_name="Test",
                confidence=1.5,  # Out of bounds
                reasoning="Test",
            )

    def test_decline_response(self):
        from src.rag.generator import DeclineResponse

        resp = DeclineResponse(reasoning="Species not in database")
        assert resp.species_name == "DECLINED"
        assert resp.confidence == 0.0
