"""
LLM Generator
Handles LLM inference for generating species identification responses.
Supports dual backends: Ollama (local) and Groq (API).
Includes structured output validation with Pydantic and retry logic.
"""

import json
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─── Response Schema ───────────────────────────────────────────────


class AlternativeSpecies(BaseModel):
    name: str = ""
    confidence: float = 0.0
    reason: str = ""


class IdentificationResponse(BaseModel):
    """Structured response from the wildlife identification LLM."""

    species_name: str = Field(description="Common name of the identified species")
    scientific_name: str = Field(default="", description="Scientific name")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(description="Why this species matches the description")
    key_features_matched: list[str] = Field(default_factory=list)
    habitat_match: str = Field(default="")
    conservation_status: str = Field(default="")
    geographic_match: bool = Field(default=True)
    cited_sources: list[str] = Field(default_factory=list, description="Chunk IDs used")
    alternative_species: list[AlternativeSpecies] = Field(default_factory=list)


class DeclineResponse(BaseModel):
    """Response when the system declines to identify."""

    species_name: str = "DECLINED"
    confidence: float = 0.0
    reasoning: str = Field(description="Why identification was declined")
    cited_sources: list[str] = Field(default_factory=list)


# ─── Prompt Builder ────────────────────────────────────────────────


def load_prompt_template(version: int = 1) -> dict:
    """Load a versioned prompt template."""
    prompt_file = Path(f"prompts/v{version}.yaml")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    with open(prompt_file) as f:
        return yaml.safe_load(f)


def build_prompt(
    query: str,
    chunks: list[dict],
    location: str | None = None,
    season: str | None = None,
    prompt_version: int = 1,
) -> tuple[str, str]:
    """
    Build system and user prompts from template + context.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    template = load_prompt_template(prompt_version)

    # Format chunks for context
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", chunk.get("raw_text", ""))
        chunk_texts.append(f"[Source: {chunk_id}]\n{text}")

    chunks_str = "\n\n---\n\n".join(chunk_texts)

    # Location and season context
    location_context = f"LOCATION: {location}" if location else ""
    season_context = f"SEASON: {season}" if season else ""

    # Fill template
    user_prompt = template["user_prompt_template"].format(
        query=query,
        chunks=chunks_str,
        location_context=location_context,
        season_context=season_context,
    )

    return template["system_prompt"], user_prompt


# ─── LLM Backends ─────────────────────────────────────────────────


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    base_url: str = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 60,
) -> str:
    """Call local Ollama for inference."""
    import ollama

    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    # Set host if custom
    base = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    client = ollama.Client(host=base)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    return response["message"]["content"]


def call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Call Groq API for fast inference."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .env file.")

    client = Groq(api_key=api_key, timeout=60.0)
    model = model or "llama-3.1-8b-instant"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content


# ─── Generator Class ──────────────────────────────────────────────


class Generator:
    """Generates species identification responses using LLM."""

    def __init__(self, config_path: str = "config/retrieval.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.llm_config = config["llm"]
        self.inference_mode = os.getenv("INFERENCE_MODE", "groq")
        self.temperature = self.llm_config["temperature"]
        self.max_tokens = self.llm_config["max_tokens"]
        self.timeout = self.llm_config["timeout_seconds"]
        self.max_retries = 2

        logger.info(f"Generator initialized. Mode: {self.inference_mode}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Route to the appropriate LLM backend."""
        if self.inference_mode == "local":
            return call_ollama(
                system_prompt,
                user_prompt,
                model=self.llm_config["local_model"],
                base_url=self.llm_config["local_base_url"],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        elif self.inference_mode == "groq":
            return call_groq(
                system_prompt,
                user_prompt,
                model=self.llm_config["groq_model"],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown inference mode: {self.inference_mode}")

    def _parse_response(self, raw_response: str) -> IdentificationResponse:
        """Parse and validate LLM response into structured format."""
        # Try to extract JSON from the response
        text = raw_response.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        json_str = text[start:end]
        data = json.loads(json_str)

        return IdentificationResponse(**data)

    def generate(
        self,
        query: str,
        chunks: list[dict],
        location: str | None = None,
        season: str | None = None,
        prompt_version: int = 1,
    ) -> dict:
        """
        Generate a species identification response.

        Args:
            query: User's natural language query
            chunks: Re-ranked context chunks
            location: Optional location context
            season: Optional season context
            prompt_version: Prompt template version to use

        Returns:
            Dict with structured response, raw response, and timing info
        """
        # Build prompts
        system_prompt, user_prompt = build_prompt(
            query=query,
            chunks=chunks,
            location=location,
            season=season,
            prompt_version=prompt_version,
        )

        # Call LLM with retry
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                raw_response = self._call_llm(system_prompt, user_prompt)
                llm_latency = time.time() - start_time

                # Parse response
                parsed = self._parse_response(raw_response)

                return {
                    "response": parsed.model_dump(),
                    "raw_response": raw_response,
                    "inference_mode": self.inference_mode,
                    "prompt_version": prompt_version,
                    "llm_latency_seconds": round(llm_latency, 3),
                    "attempt": attempt + 1,
                    "chunks_used": len(chunks),
                }

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}: {last_error}")

        # All retries failed — return decline response
        logger.error(f"All {self.max_retries + 1} attempts failed. Last error: {last_error}")
        decline = DeclineResponse(
            reasoning=f"System failed to generate a valid response after {self.max_retries + 1} attempts. Error: {last_error}",
        )
        return {
            "response": decline.model_dump(),
            "raw_response": "",
            "inference_mode": self.inference_mode,
            "prompt_version": prompt_version,
            "llm_latency_seconds": 0,
            "attempt": self.max_retries + 1,
            "chunks_used": len(chunks),
            "error": last_error,
        }
