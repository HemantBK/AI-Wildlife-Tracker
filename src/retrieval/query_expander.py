"""
Query Expander (Feature Engineering for RAG)
Expands user queries with synonyms, habitat terms, and extracts
structured entities (location, season, features) for better retrieval.
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Habitat synonym dictionary for query expansion
HABITAT_SYNONYMS = {
    "water": ["aquatic", "wetland", "riparian", "lake", "river", "pond", "stream", "marsh"],
    "forest": ["woodland", "jungle", "deciduous", "tropical", "canopy", "understory"],
    "grassland": ["savanna", "meadow", "prairie", "scrubland", "open country"],
    "mountain": ["highland", "alpine", "hill", "elevation", "montane"],
    "desert": ["arid", "dry", "sand", "semi-arid", "thar"],
    "coast": ["coastal", "marine", "beach", "shore", "mangrove", "estuary", "sea"],
    "urban": ["city", "town", "garden", "park", "residential"],
    "farm": ["agricultural", "farmland", "crop", "field", "cultivation"],
    "swamp": ["marshland", "bog", "wetland", "backwater"],
}

# Color variations
COLOR_SYNONYMS = {
    "brown": ["rufous", "chestnut", "tawny", "russet", "umber"],
    "blue": ["azure", "cobalt", "cerulean", "turquoise", "indigo"],
    "green": ["olive", "emerald", "viridian", "lime"],
    "red": ["crimson", "scarlet", "vermilion", "ruby"],
    "yellow": ["golden", "saffron", "amber", "ochre"],
    "black": ["dark", "ebony", "jet"],
    "white": ["pale", "ivory", "snowy", "albino"],
    "orange": ["tangerine", "flame", "rust"],
    "grey": ["gray", "silver", "ashen", "slate"],
}

# Size mappings
SIZE_TERMS = {
    "tiny": ["very small", "miniature", "diminutive"],
    "small": ["little", "compact", "petite"],
    "medium": ["moderate", "mid-sized"],
    "large": ["big", "sizeable", "substantial"],
    "huge": ["massive", "enormous", "giant", "very large"],
}

# Indian state name patterns
INDIAN_STATES = [
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "andaman",
    "nicobar",
    "ladakh",
    "jammu",
    "kashmir",
    "delhi",
    "chandigarh",
]

# Notable Indian wildlife locations
WILDLIFE_LOCATIONS = {
    "ranthambore": "Rajasthan",
    "jim corbett": "Uttarakhand",
    "corbett": "Uttarakhand",
    "kaziranga": "Assam",
    "sundarbans": "West Bengal",
    "gir": "Gujarat",
    "periyar": "Kerala",
    "bandipur": "Karnataka",
    "kanha": "Madhya Pradesh",
    "pench": "Madhya Pradesh",
    "tadoba": "Maharashtra",
    "nagarhole": "Karnataka",
    "bharatpur": "Rajasthan",
    "keoladeo": "Rajasthan",
    "chilika": "Odisha",
    "western ghats": "Kerala",
    "nilgiris": "Tamil Nadu",
    "valley of flowers": "Uttarakhand",
    "hemis": "Ladakh",
    "dachigam": "Jammu and Kashmir",
}

# Season patterns
SEASON_KEYWORDS = {
    "summer": ["summer", "hot", "may", "june", "april"],
    "monsoon": ["monsoon", "rainy", "rain", "july", "august", "september", "wet"],
    "winter": ["winter", "cold", "december", "january", "february", "cool"],
    "spring": ["spring", "march", "bloom"],
    "autumn": ["autumn", "fall", "october", "november"],
}


def extract_location(query: str) -> str | None:
    """Extract location/state from query."""
    query_lower = query.lower()

    # Check wildlife parks first (more specific)
    for park, state in WILDLIFE_LOCATIONS.items():
        if park in query_lower:
            return state

    # Check state names
    for state in INDIAN_STATES:
        if state in query_lower:
            return state.title()

    return None


def extract_season(query: str) -> str | None:
    """Extract season from query."""
    query_lower = query.lower()
    for season, keywords in SEASON_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return season
    return None


def extract_features(query: str) -> dict:
    """Extract descriptive features from query."""
    query_lower = query.lower()
    features = {
        "colors": [],
        "size": None,
        "habitat_hints": [],
    }

    # Extract colors
    for color in COLOR_SYNONYMS:
        if color in query_lower:
            features["colors"].append(color)

    # Extract size
    for size, synonyms in SIZE_TERMS.items():
        if size in query_lower or any(s in query_lower for s in synonyms):
            features["size"] = size
            break

    # Extract habitat hints
    for habitat in HABITAT_SYNONYMS:
        if habitat in query_lower or any(s in query_lower for s in HABITAT_SYNONYMS[habitat]):
            features["habitat_hints"].append(habitat)

    return features


def expand_query(query: str) -> str:
    """
    Expand query with synonym terms for better retrieval.

    Example:
        "brown bird near water" →
        "brown rufous chestnut tawny bird near water aquatic wetland riparian lake river"
    """
    query_lower = query.lower()
    expansions = []

    # Expand habitat terms
    for habitat, synonyms in HABITAT_SYNONYMS.items():
        if habitat in query_lower:
            expansions.extend(synonyms[:3])  # Add top 3 synonyms

    # Expand color terms
    for color, synonyms in COLOR_SYNONYMS.items():
        if color in query_lower:
            expansions.extend(synonyms[:2])  # Add top 2 synonyms

    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        logger.debug(f"Query expanded: '{query}' → '{expanded}'")
        return expanded

    return query


def preprocess_query(query: str) -> dict:
    """
    Full query preprocessing: extract entities, expand terms.

    Returns:
        Dict with original query, expanded query, extracted location/season/features
    """
    location = extract_location(query)
    season = extract_season(query)
    features = extract_features(query)
    expanded = expand_query(query)

    return {
        "original_query": query,
        "expanded_query": expanded,
        "location": location,
        "season": season,
        "features": features,
    }


if __name__ == "__main__":
    # Test query preprocessing
    test_queries = [
        "small brown bird near water in Kerala",
        "large striped cat Ranthambore summer",
        "venomous snake with hood in monsoon",
        "blue feathered bird dancing in rain Western Ghats",
        "huge grey animal in forest Kaziranga",
    ]

    for q in test_queries:
        result = preprocess_query(q)
        print(f"\nQuery: {q}")
        print(f"  Expanded: {result['expanded_query']}")
        print(f"  Location: {result['location']}")
        print(f"  Season:   {result['season']}")
        print(f"  Features: {result['features']}")
