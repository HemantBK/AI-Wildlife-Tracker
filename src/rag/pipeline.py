"""
Main RAG Pipeline
Orchestrates the full query path:
  Query → Preprocessing → Hybrid Search → Re-ranking → Geographic Filter → LLM → Validated Response

Every step is timed for observability.
Optionally emits Langfuse traces for each pipeline run.
"""

import logging
import time
import uuid

import yaml

from src.monitoring.tracing import (
    flush as flush_traces,
)
from src.monitoring.tracing import (
    get_current_trace_url,
    score_trace,
    set_trace_output,
    traced_generation,
    traced_pipeline,
    traced_span,
)
from src.rag.generator import Generator
from src.rag.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_expander import preprocess_query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WildlifeRAGPipeline:
    """
    End-to-end RAG pipeline for wildlife identification.

    Flow:
    1. Query preprocessing (entity extraction, expansion)
    2. Hybrid search (vector + BM25 → RRF fusion)
    3. Cross-encoder re-ranking
    4. Geographic filtering
    5. LLM generation with structured output
    6. Response validation
    """

    def __init__(self, config_path: str = "config/retrieval.yaml"):
        logger.info("Initializing Wildlife RAG Pipeline...")
        start = time.time()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.searcher = HybridSearcher(config_path)
        self.reranker = Reranker(config_path)
        self.generator = Generator(config_path)

        self.chunk_count_for_llm = self.config["llm"]["chunk_count_for_llm"]
        self.geo_filter_enabled = self.config["geographic"]["enabled"]
        self.geo_strict = self.config["geographic"]["strict"]

        elapsed = time.time() - start
        logger.info(f"Pipeline ready in {elapsed:.1f}s")

    def _apply_geographic_filter(self, chunks: list[dict], location: str | None) -> list[dict]:
        """Filter or boost chunks based on geographic relevance."""
        if not location or not self.geo_filter_enabled:
            return chunks

        location_lower = location.lower()
        geo_relevant = []
        geo_other = []

        for chunk in chunks:
            regions = chunk.get("metadata", {}).get("geographic_regions", "")
            if location_lower in regions.lower():
                chunk["geo_match"] = True
                geo_relevant.append(chunk)
            else:
                chunk["geo_match"] = False
                geo_other.append(chunk)

        if self.geo_strict:
            # Only return geographically relevant chunks
            logger.info(
                f"Geo filter (strict): {len(geo_relevant)} relevant, {len(geo_other)} filtered out"
            )
            return geo_relevant if geo_relevant else chunks  # Fallback to all if none match
        else:
            # Boost relevant chunks by putting them first
            logger.info(
                f"Geo filter (boost): {len(geo_relevant)} boosted, {len(geo_other)} de-prioritized"
            )
            return geo_relevant + geo_other

    def identify(
        self,
        query: str,
        location: str | None = None,
        season: str | None = None,
        prompt_version: int = 1,
    ) -> dict:
        """
        Run the full identification pipeline.

        Args:
            query: Natural language description of the sighting
            location: Optional location (state, park, city)
            season: Optional season

        Returns:
            Complete result dict with response, timings, and trace data
        """
        request_id = str(uuid.uuid4())[:8]
        pipeline_start = time.time()
        timings = {}

        logger.info(f"[{request_id}] Query: {query}")

        # ── Langfuse-traced pipeline (no-op if not configured) ─
        with traced_pipeline(request_id, query, location, season):
            return self._run_pipeline(
                request_id,
                query,
                location,
                season,
                prompt_version,
                pipeline_start,
                timings,
            )

    def _run_pipeline(
        self,
        request_id: str,
        query: str,
        location: str | None,
        season: str | None,
        prompt_version: int,
        pipeline_start: float,
        timings: dict,
    ) -> dict:
        """Inner pipeline logic wrapped by Langfuse trace context."""

        # ── Step 1: Query Preprocessing ────────────────────────
        with traced_span("query_preprocessing", input_data={"query": query}) as span:
            t0 = time.time()
            query_data = preprocess_query(query)

            # Use extracted location/season if not explicitly provided
            if not location:
                location = query_data.get("location")
            if not season:
                season = query_data.get("season")

            expanded_query = query_data["expanded_query"]
            timings["preprocessing_ms"] = round((time.time() - t0) * 1000, 1)
            logger.info(f"[{request_id}] Location: {location}, Season: {season}")

            span.update(
                output={"expanded_query": expanded_query, "location": location, "season": season},
                metadata={"latency_ms": str(timings["preprocessing_ms"])},
            )

        # ── Step 2: Hybrid Search ──────────────────────────────
        with traced_span(
            "hybrid_search", input_data={"query": expanded_query, "geo_filter": location or ""}
        ) as span:
            t0 = time.time()
            search_results = self.searcher.search(
                query=expanded_query,
                geographic_filter=location,
            )
            timings["hybrid_search_ms"] = round((time.time() - t0) * 1000, 1)
            logger.info(f"[{request_id}] Hybrid search: {len(search_results)} candidates")

            span.update(
                output={"result_count": len(search_results)},
                metadata={"latency_ms": str(timings["hybrid_search_ms"])},
            )

        # ── Step 3: Re-ranking ─────────────────────────────────
        with traced_span("reranking", input_data={"candidate_count": len(search_results)}) as span:
            t0 = time.time()
            reranked = self.reranker.rerank(
                query=query,  # Use original query for re-ranking (more precise)
                candidates=search_results,
            )
            timings["reranking_ms"] = round((time.time() - t0) * 1000, 1)
            logger.info(f"[{request_id}] Re-ranked: {len(reranked)} results")

            span.update(
                output={"result_count": len(reranked)},
                metadata={"latency_ms": str(timings["reranking_ms"])},
            )

        # ── Step 4: Geographic Filter ──────────────────────────
        with traced_span(
            "geographic_filter",
            input_data={"location": location or "", "input_count": len(reranked)},
        ) as span:
            t0 = time.time()
            filtered = self._apply_geographic_filter(reranked, location)
            timings["geo_filter_ms"] = round((time.time() - t0) * 1000, 1)

            span.update(
                output={"output_count": len(filtered)},
                metadata={"latency_ms": str(timings["geo_filter_ms"])},
            )

        # ── Step 5: Select top chunks for LLM ─────────────────
        top_chunks = filtered[: self.chunk_count_for_llm]

        if not top_chunks:
            # No relevant chunks found — decline gracefully
            total_time = time.time() - pipeline_start
            decline_result = {
                "request_id": request_id,
                "query": query,
                "response": {
                    "species_name": "DECLINED",
                    "confidence": 0.0,
                    "reasoning": "No relevant information found in the knowledge base for this query.",
                    "cited_sources": [],
                },
                "location": location,
                "season": season,
                "timings": timings,
                "total_latency_seconds": round(total_time, 3),
                "chunks_retrieved": 0,
                "chunks_used": 0,
            }
            # Score the declined trace
            score_trace("confidence", 0.0)
            score_trace("identified", 0.0, comment="DECLINED - no chunks found")
            set_trace_output(decline_result["response"])
            flush_traces()
            return decline_result

        # ── Step 6: LLM Generation ────────────────────────────
        with traced_generation(
            "llm_generation",
            model=f"groq/{self.config.get('llm', {}).get('model', 'unknown')}",
            input_data={"query": query, "chunks_used": len(top_chunks)},
        ) as gen:
            t0 = time.time()
            generation_result = self.generator.generate(
                query=query,
                chunks=top_chunks,
                location=location,
                season=season,
                prompt_version=prompt_version,
            )
            timings["llm_generation_ms"] = round((time.time() - t0) * 1000, 1)

            gen.update(
                output=generation_result.get("raw_response", "")[:500],
                metadata={
                    "latency_ms": str(timings["llm_generation_ms"]),
                    "attempt": str(generation_result.get("attempt", 1)),
                    "inference_mode": generation_result.get("inference_mode", ""),
                },
            )

        # ── Step 7: Confidence check (decline low confidence) ──
        response = generation_result["response"]
        confidence = response.get("confidence", 0)
        if confidence < 0.3 and response.get("species_name") != "DECLINED":
            logger.warning(f"[{request_id}] Low confidence ({confidence}), marking as uncertain")
            response["reasoning"] = f"[LOW CONFIDENCE] {response.get('reasoning', '')}"

        # ── Assemble final result ──────────────────────────────
        total_time = time.time() - pipeline_start
        timings["total_ms"] = round(total_time * 1000, 1)

        species_name = response.get("species_name", "DECLINED")

        # Score the trace
        score_trace("confidence", confidence)
        score_trace("identified", 1.0 if species_name != "DECLINED" else 0.0)
        score_trace("latency", round(total_time, 3), comment=f"{total_time:.2f}s total pipeline")

        # Set trace-level output
        set_trace_output(
            {
                "species_name": species_name,
                "scientific_name": response.get("scientific_name", ""),
                "confidence": confidence,
                "reasoning": response.get("reasoning", "")[:500],
            }
        )

        # Get trace URL for the result
        trace_url = get_current_trace_url()

        result = {
            "request_id": request_id,
            "query": query,
            "response": response,
            "location": location,
            "season": season,
            "inference_mode": generation_result.get("inference_mode"),
            "prompt_version": prompt_version,
            "timings": timings,
            "total_latency_seconds": round(total_time, 3),
            "chunks_retrieved": len(search_results),
            "chunks_after_rerank": len(reranked),
            "chunks_used": len(top_chunks),
            "trace_url": trace_url,
            "retrieval_details": {
                "expanded_query": expanded_query,
                "features_extracted": query_data.get("features", {}),
                "top_chunk_scores": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "species": c.get("metadata", {}).get("species_name"),
                        "reranker_score": round(c.get("reranker_score", 0), 4),
                        "rrf_score": round(c.get("rrf_score", 0), 4),
                        "geo_match": c.get("geo_match"),
                    }
                    for c in top_chunks
                ],
            },
        }

        flush_traces()

        logger.info(
            f"[{request_id}] Result: {species_name} (confidence: {confidence}) in {total_time:.2f}s"
        )

        return result


def main():
    """Interactive test of the RAG pipeline."""
    pipeline = WildlifeRAGPipeline()

    print("\n" + "=" * 60)
    print("  Wildlife Tracker - RAG Pipeline Test")
    print("=" * 60)

    test_queries = [
        {
            "query": "I saw a large orange and black striped cat in the forest",
            "location": "Madhya Pradesh",
        },
        {"query": "small bright blue bird with long tail feathers", "location": "Kerala"},
        {"query": "large grey animal with one horn near a river", "location": "Assam"},
        {"query": "colorful bird with a fan-shaped crest that dances in the rain"},
        {"query": "polar bear in Tamil Nadu"},  # Trick query
    ]

    for test in test_queries:
        print(f"\n{'─' * 60}")
        result = pipeline.identify(**test)
        resp = result["response"]
        print(f"Query:      {result['query']}")
        print(f"Location:   {result.get('location', 'N/A')}")
        print(f"Species:    {resp.get('species_name', 'N/A')}")
        print(f"Confidence: {resp.get('confidence', 0)}")
        print(f"Reasoning:  {resp.get('reasoning', '')[:150]}")
        print(f"Latency:    {result['total_latency_seconds']}s")
        print(f"Timings:    {result['timings']}")


if __name__ == "__main__":
    main()
