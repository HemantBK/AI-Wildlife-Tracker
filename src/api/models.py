"""
API Request/Response Models
Pydantic schemas for all API endpoints.
Separates API contracts from internal pipeline models.
"""

from pydantic import BaseModel, Field

# ─── Request Models ──────────────────────────────────────────────


class IdentifyRequest(BaseModel):
    """POST /identify request body."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language description of the wildlife sighting",
        examples=["I saw a large orange and black striped cat in the forest"],
    )
    location: str | None = Field(
        default=None,
        max_length=200,
        description="Location of sighting (state, park, city)",
        examples=["Madhya Pradesh", "Ranthambore National Park"],
    )
    season: str | None = Field(
        default=None,
        description="Season of sighting",
        examples=["monsoon", "winter", "summer"],
    )
    prompt_version: int = Field(
        default=1,
        ge=1,
        description="Prompt template version to use",
    )


class FeedbackRequest(BaseModel):
    """POST /feedback request body."""

    request_id: str = Field(
        ...,
        min_length=1,
        description="The request_id from the /identify response",
    )
    correct_species: str | None = Field(
        default=None,
        description="The actual correct species (if different from prediction)",
    )
    was_correct: bool = Field(
        ...,
        description="Whether the identification was correct",
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Additional feedback notes",
    )


# ─── Response Models ─────────────────────────────────────────────


class AlternativeSpeciesResponse(BaseModel):
    """An alternative species suggestion."""

    name: str = ""
    confidence: float = 0.0
    reason: str = ""


class IdentifyResponse(BaseModel):
    """POST /identify response body."""

    request_id: str
    query: str
    species_name: str
    scientific_name: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    key_features_matched: list[str] = []
    habitat_match: str = ""
    conservation_status: str = ""
    geographic_match: bool = True
    cited_sources: list[str] = []
    alternative_species: list[AlternativeSpeciesResponse] = []
    location: str | None = None
    season: str | None = None
    inference_mode: str | None = None
    total_latency_seconds: float = 0.0
    chunks_retrieved: int = 0
    chunks_used: int = 0
    # Vision fields (only populated for image-based identification)
    input_mode: str = "text"
    vision_description: str | None = None
    vision_latency_seconds: float | None = None
    vision_model: str | None = None


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str
    components: dict = Field(default_factory=dict)


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # "ok" or "error"
    message: str = ""
    latency_ms: float | None = None


class MetricsResponse(BaseModel):
    """GET /metrics response body."""

    total_requests: int = 0
    successful_identifications: int = 0
    declined_identifications: int = 0
    error_count: int = 0
    avg_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    avg_confidence: float = 0.0
    uptime_seconds: float = 0.0
    feedback_count: int = 0
    accuracy_from_feedback: float | None = None
    top_species: list[dict] = []
    requests_by_hour: list[dict] = []


class FeedbackResponse(BaseModel):
    """POST /feedback response body."""

    status: str = "received"
    feedback_id: str
    message: str = "Thank you for your feedback!"


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
    request_id: str | None = None
