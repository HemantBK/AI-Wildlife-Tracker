"""
Benchmark Queries for Model Comparison
30 queries across difficulty levels, each with expected species answer.
Used by embedding comparison, LLM comparison, and all Phase 3 experiments.
"""

# Each query has: query text, expected species, difficulty, optional location
BENCHMARK_QUERIES = [
    # ── EASY (12 queries) — straightforward, single species ──────────
    {
        "query": "What does a peacock look like?",
        "expected_species": "Indian Peafowl",
        "expected_scientific": "Pavo cristatus",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Tell me about the Bengal Tiger",
        "expected_species": "Bengal Tiger",
        "expected_scientific": "Panthera tigris tigris",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Describe the King Cobra",
        "expected_species": "King Cobra",
        "expected_scientific": "Ophiophagus hannah",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "What is the Indian Elephant?",
        "expected_species": "Indian Elephant",
        "expected_scientific": "Elephas maximus indicus",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Tell me about the Indian Rhinoceros",
        "expected_species": "Indian Rhinoceros",
        "expected_scientific": "Rhinoceros unicornis",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Describe the Common Kingfisher found in India",
        "expected_species": "Common Kingfisher",
        "expected_scientific": "Alcedo atthis",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "What does a Sloth Bear look like?",
        "expected_species": "Sloth Bear",
        "expected_scientific": "Melursus ursinus",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Describe the Indian Cobra",
        "expected_species": "Indian Cobra",
        "expected_scientific": "Naja naja",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Tell me about the Gharial crocodile",
        "expected_species": "Gharial",
        "expected_scientific": "Gavialis gangeticus",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "What is a Blackbuck?",
        "expected_species": "Blackbuck",
        "expected_scientific": "Antilope cervicapra",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "Describe the Indian Star Tortoise",
        "expected_species": "Indian Star Tortoise",
        "expected_scientific": "Geochelone elegans",
        "difficulty": "easy",
        "location": None,
    },
    {
        "query": "What does a Rose-ringed Parakeet look like?",
        "expected_species": "Rose-ringed Parakeet",
        "expected_scientific": "Psittacula krameri",
        "difficulty": "easy",
        "location": None,
    },
    # ── MEDIUM (9 queries) — description-based, needs feature matching ─
    {
        "query": "Small green bird with a red beak near a river in Kerala",
        "expected_species": "White-throated Kingfisher",
        "expected_scientific": "Halcyon smyrnensis",
        "difficulty": "medium",
        "location": "Kerala",
    },
    {
        "query": "Large grey animal with one horn near a river",
        "expected_species": "Indian Rhinoceros",
        "expected_scientific": "Rhinoceros unicornis",
        "difficulty": "medium",
        "location": "Assam",
    },
    {
        "query": "Orange and black striped large cat in the forest of Madhya Pradesh",
        "expected_species": "Bengal Tiger",
        "expected_scientific": "Panthera tigris tigris",
        "difficulty": "medium",
        "location": "Madhya Pradesh",
    },
    {
        "query": "Bright blue bird with a long tail seen in Western Ghats",
        "expected_species": "Indian Roller",
        "expected_scientific": "Coracias benghalensis",
        "difficulty": "medium",
        "location": "Kerala",
    },
    {
        "query": "Spotted deer with white dots on brown coat grazing in an open meadow",
        "expected_species": "Chital",
        "expected_scientific": "Axis axis",
        "difficulty": "medium",
        "location": None,
    },
    {
        "query": "Black bear with a white V-shaped chest mark seen climbing trees",
        "expected_species": "Sloth Bear",
        "expected_scientific": "Melursus ursinus",
        "difficulty": "medium",
        "location": None,
    },
    {
        "query": "Large wading bird with pink feathers standing on one leg in a lake",
        "expected_species": "Greater Flamingo",
        "expected_scientific": "Phoenicopterus roseus",
        "difficulty": "medium",
        "location": "Gujarat",
    },
    {
        "query": "Small nocturnal owl with spots on its head seen near a village",
        "expected_species": "Spotted Owlet",
        "expected_scientific": "Athene brama",
        "difficulty": "medium",
        "location": None,
    },
    {
        "query": "Wild dog with reddish-brown fur hunting in a pack in Karnataka",
        "expected_species": "Indian Wild Dog",
        "expected_scientific": "Cuon alpinus",
        "difficulty": "medium",
        "location": "Karnataka",
    },
    # ── HARD (5 queries) — distinguishing similar species ────────────
    {
        "query": "Difference between Indian Robin and Oriental Magpie-Robin",
        "expected_species": "Oriental Magpie-Robin",
        "expected_scientific": "Copsychus saularis",
        "difficulty": "hard",
        "location": None,
    },
    {
        "query": "A brown snake with chain-like pattern found in farmland at dusk",
        "expected_species": "Russell's Viper",
        "expected_scientific": "Daboia russelii",
        "difficulty": "hard",
        "location": None,
    },
    {
        "query": "Small black and white bird with a long forked tail perched on a wire",
        "expected_species": "Black Drongo",
        "expected_scientific": "Dicrurus macrocercus",
        "difficulty": "hard",
        "location": None,
    },
    {
        "query": "Large brown eagle with a crest soaring over a hill station",
        "expected_species": "Crested Serpent Eagle",
        "expected_scientific": "Spilornis cheela",
        "difficulty": "hard",
        "location": "Himachal Pradesh",
    },
    {
        "query": "Grey langur with a black face sitting on a temple wall",
        "expected_species": "Hanuman Langur",
        "expected_scientific": "Semnopithecus entellus",
        "difficulty": "hard",
        "location": "Rajasthan",
    },
    # ── TRICK (4 queries) — should decline or flag ───────────────────
    {
        "query": "Polar bear spotted in Tamil Nadu",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "difficulty": "trick",
        "location": "Tamil Nadu",
    },
    {
        "query": "What is the best pizza restaurant near me?",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "difficulty": "trick",
        "location": None,
    },
    {
        "query": "Tell me about the Dodo bird in India",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "difficulty": "trick",
        "location": None,
    },
    {
        "query": "Unicorn spotted in Ranthambore National Park",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "difficulty": "trick",
        "location": "Rajasthan",
    },
]


def get_queries_by_difficulty(difficulty: str = None) -> list[dict]:
    """Get benchmark queries filtered by difficulty."""
    if difficulty is None:
        return BENCHMARK_QUERIES
    return [q for q in BENCHMARK_QUERIES if q["difficulty"] == difficulty]


def get_query_stats() -> dict:
    """Get stats about the benchmark query set."""
    from collections import Counter

    difficulties = Counter(q["difficulty"] for q in BENCHMARK_QUERIES)
    with_location = sum(1 for q in BENCHMARK_QUERIES if q["location"])
    return {
        "total": len(BENCHMARK_QUERIES),
        "by_difficulty": dict(difficulties),
        "with_location": with_location,
    }


if __name__ == "__main__":
    stats = get_query_stats()
    print(f"Benchmark Queries: {stats['total']}")
    print(f"By difficulty: {stats['by_difficulty']}")
    print(f"With location: {stats['with_location']}")
