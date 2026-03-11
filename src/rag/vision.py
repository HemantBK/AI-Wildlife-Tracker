"""
Vision Module — Multimodal Wildlife Identification

Uses Groq's vision-capable LLM to analyze uploaded wildlife photos.
Two modes:
  1. Vision-only: Directly identify species from image
  2. Vision + RAG (hybrid): Vision model describes the image → feeds into RAG pipeline

The hybrid approach is more accurate because it grounds the identification
in the curated knowledge base rather than relying solely on the vision model.
"""

import base64
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Image Utilities ──────────────────────────────────────────────


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_bytes_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(filename: str) -> str:
    """Determine MIME type from filename extension."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


# ─── Vision Analysis ─────────────────────────────────────────────


VISION_DESCRIBE_PROMPT = """You are a wildlife biology expert. Analyze this image and describe the animal you see.

Focus on:
1. Physical features: size, color, patterns, distinctive markings
2. Body parts: beak shape, tail, horns, wings, fur/feathers
3. Behavior: what the animal is doing
4. Habitat clues: what environment/vegetation is visible

Provide a detailed natural language description that could be used to identify the species.
Do NOT guess the species name — just describe what you observe.

Format your response as a single descriptive paragraph, like a field naturalist would write in their observation notes."""


VISION_IDENTIFY_PROMPT = """You are an expert wildlife biologist specializing in Indian fauna.

Analyze this image and identify the species. Respond in this exact JSON format:
{
    "species_name": "Common name of the species",
    "scientific_name": "Scientific binomial name",
    "confidence": 0.85,
    "description": "Detailed description of what you see in the image",
    "key_features": ["feature1", "feature2", "feature3"],
    "habitat_clues": "What the environment in the image suggests",
    "reasoning": "Why you identified this species based on visible features"
}

If you cannot identify the species or the image doesn't contain a clear animal, set species_name to "UNKNOWN" and confidence to 0.0."""


def analyze_image_with_groq(
    image_base64: str,
    mime_type: str = "image/jpeg",
    mode: str = "describe",
    location: str = None,
    season: str = None,
) -> dict:
    """
    Analyze a wildlife image using Groq's vision model.

    Args:
        image_base64: Base64-encoded image data
        mime_type: Image MIME type (image/jpeg, image/png, etc.)
        mode: "describe" (for RAG pipeline) or "identify" (direct identification)
        location: Optional location context
        season: Optional season context

    Returns:
        dict with 'description' (describe mode) or full identification (identify mode)
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env file.")

    client = Groq(api_key=api_key, timeout=60.0)

    # Choose prompt based on mode
    if mode == "describe":
        system_prompt = VISION_DESCRIBE_PROMPT
        if location or season:
            context = []
            if location:
                context.append(f"Location: {location}")
            if season:
                context.append(f"Season: {season}")
            system_prompt += f"\n\nContext: {', '.join(context)}"
    else:
        system_prompt = VISION_IDENTIFY_PROMPT
        if location:
            system_prompt += f"\n\nThe sighting was in: {location}"
        if season:
            system_prompt += f"\n\nSeason: {season}"

    # Build the vision message
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}",
                },
            },
            {
                "type": "text",
                "text": (
                    "Analyze this wildlife image."
                    if mode == "describe"
                    else "Identify the species in this image. Respond in JSON."
                ),
            },
        ],
    }

    logger.info(f"Calling Groq vision model (mode={mode})...")
    start = time.time()

    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    elapsed = time.time() - start
    content = response.choices[0].message.content
    logger.info(f"Vision analysis complete in {elapsed:.1f}s")

    if mode == "describe":
        return {
            "description": content.strip(),
            "vision_latency_seconds": round(elapsed, 3),
            "vision_model": "llama-3.2-90b-vision-preview",
        }
    else:
        # Try to parse JSON response
        import json

        try:
            # Find JSON in the response (model sometimes wraps it)
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                parsed["vision_latency_seconds"] = round(elapsed, 3)
                parsed["vision_model"] = "llama-3.2-90b-vision-preview"
                return parsed
        except json.JSONDecodeError:
            pass

        return {
            "species_name": "UNKNOWN",
            "confidence": 0.0,
            "description": content.strip(),
            "reasoning": "Could not parse structured response from vision model",
            "vision_latency_seconds": round(elapsed, 3),
            "vision_model": "llama-3.2-90b-vision-preview",
        }


# ─── Hybrid Vision + RAG Pipeline ────────────────────────────────


def vision_rag_identify(
    image_base64: str,
    mime_type: str = "image/jpeg",
    location: str = None,
    season: str = None,
) -> dict:
    """
    Hybrid multimodal identification: Vision → Description → RAG Pipeline.

    Flow:
    1. Vision model analyzes image → produces natural language description
    2. Description is fed into the existing RAG pipeline
    3. RAG pipeline identifies species using the knowledge base

    This gives more accurate results than vision-only because identifications
    are grounded in curated species data rather than just the vision model's training.

    Args:
        image_base64: Base64-encoded image data
        mime_type: Image MIME type
        location: Optional location for geographic filtering
        season: Optional season context

    Returns:
        Full identification result with both vision and RAG metadata
    """
    total_start = time.time()

    # Step 1: Vision model describes the image
    logger.info("Step 1: Analyzing image with vision model...")
    vision_result = analyze_image_with_groq(
        image_base64=image_base64,
        mime_type=mime_type,
        mode="describe",
        location=location,
        season=season,
    )

    description = vision_result["description"]
    logger.info(f"Vision description: {description[:100]}...")

    # Step 2: Feed description into RAG pipeline
    logger.info("Step 2: Running RAG pipeline on vision description...")
    from src.rag.pipeline import WildlifeRAGPipeline

    pipeline = WildlifeRAGPipeline()
    rag_result = pipeline.identify(
        query=description,
        location=location,
        season=season,
    )

    # Combine results
    total_time = time.time() - total_start

    combined = {
        **rag_result,
        "input_mode": "image",
        "vision_description": description,
        "vision_latency_seconds": vision_result["vision_latency_seconds"],
        "vision_model": vision_result["vision_model"],
        "total_latency_seconds": round(total_time, 3),
    }

    logger.info(
        f"Hybrid vision+RAG complete in {total_time:.1f}s "
        f"(vision: {vision_result['vision_latency_seconds']:.1f}s, "
        f"RAG: {rag_result.get('total_latency_seconds', 0):.1f}s)"
    )

    return combined
