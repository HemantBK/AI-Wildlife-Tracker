"""
Data Cleaner
Merges data from GBIF, Wikipedia, and iNaturalist into unified species records.
Handles deduplication, normalization, and validation.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

# Location name normalization mapping
LOCATION_ALIASES = {
    "bangalore": "Karnataka",
    "bengaluru": "Karnataka",
    "mumbai": "Maharashtra",
    "bombay": "Maharashtra",
    "chennai": "Tamil Nadu",
    "madras": "Tamil Nadu",
    "kolkata": "West Bengal",
    "calcutta": "West Bengal",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "hyderabad": "Telangana",
    "pune": "Maharashtra",
    "ahmedabad": "Gujarat",
    "jaipur": "Rajasthan",
    "lucknow": "Uttar Pradesh",
    "kochi": "Kerala",
    "cochin": "Kerala",
    "thiruvananthapuram": "Kerala",
    "trivandrum": "Kerala",
    "guwahati": "Assam",
    "shimla": "Himachal Pradesh",
    "dehradun": "Uttarakhand",
    "bhopal": "Madhya Pradesh",
    "raipur": "Chhattisgarh",
    "ranchi": "Jharkhand",
    "patna": "Bihar",
    "bhubaneswar": "Odisha",
    "gangtok": "Sikkim",
    "imphal": "Manipur",
    "shillong": "Meghalaya",
    "aizawl": "Mizoram",
    "kohima": "Nagaland",
    "agartala": "Tripura",
    "itanagar": "Arunachal Pradesh",
    "panaji": "Goa",
    "port blair": "Andaman and Nicobar Islands",
    "srinagar": "Jammu and Kashmir",
    "leh": "Ladakh",
    "chandigarh": "Punjab",
}

# Conservation status normalization
IUCN_MAPPING = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
    "EW": "Extinct in the Wild",
    "EX": "Extinct",
    "least concern": "Least Concern",
    "near threatened": "Near Threatened",
    "vulnerable": "Vulnerable",
    "endangered": "Endangered",
    "critically endangered": "Critically Endangered",
}


def normalize_name(name: str) -> str:
    """Normalize a species name for deduplication."""
    return re.sub(r"[^a-z\s]", "", name.lower()).strip()


def normalize_location(location: str) -> str:
    """Normalize a location name to Indian state."""
    loc_lower = location.lower().strip()
    if loc_lower in LOCATION_ALIASES:
        return LOCATION_ALIASES[loc_lower]
    # Check if it's already a valid state name
    for state in LOCATION_ALIASES.values():
        if state.lower() == loc_lower:
            return state
    return location.strip()


def normalize_conservation_status(status: str) -> str:
    """Normalize IUCN conservation status."""
    if not status:
        return "Not Evaluated"
    status_clean = status.strip()
    return IUCN_MAPPING.get(status_clean, IUCN_MAPPING.get(status_clean.lower(), status_clean))


def load_gbif_data() -> dict[str, dict]:
    """Load and index GBIF data by normalized scientific name."""
    gbif_dir = RAW_DIR / "gbif"
    species = {}
    if not gbif_dir.exists():
        logger.warning("No GBIF data found")
        return species

    for filepath in gbif_dir.glob("*.json"):
        if filepath.name == "collection_summary.json":
            continue
        with open(filepath, encoding="utf-8") as f:
            for sp in json.load(f):
                key = normalize_name(sp.get("canonical_name", ""))
                if key:
                    species[key] = sp
    logger.info(f"Loaded {len(species)} species from GBIF")
    return species


def load_wikipedia_data() -> dict[str, dict]:
    """Load and index Wikipedia data by species name."""
    wiki_dir = RAW_DIR / "wikipedia"
    species = {}
    if not wiki_dir.exists():
        logger.warning("No Wikipedia data found")
        return species

    for filepath in wiki_dir.glob("*.json"):
        if filepath.name == "failed_species.json":
            continue
        with open(filepath, encoding="utf-8") as f:
            for sp in json.load(f):
                key = normalize_name(sp.get("scientific_name", "") or sp.get("species_name", ""))
                if key:
                    species[key] = sp
    logger.info(f"Loaded {len(species)} species from Wikipedia")
    return species


def load_inaturalist_data() -> dict[str, dict]:
    """Load and index iNaturalist data by scientific name."""
    inat_dir = RAW_DIR / "inaturalist"
    species = {}
    if not inat_dir.exists():
        logger.warning("No iNaturalist data found")
        return species

    for filepath in inat_dir.glob("*.json"):
        if filepath.name == "collection_summary.json":
            continue
        with open(filepath, encoding="utf-8") as f:
            for sp in json.load(f):
                key = normalize_name(sp.get("scientific_name", ""))
                if key:
                    species[key] = sp
    logger.info(f"Loaded {len(species)} species from iNaturalist")
    return species


def merge_species(gbif: dict, wiki: dict, inat: dict) -> dict:
    """Merge data from all three sources for a single species."""
    # Start with GBIF as the taxonomic backbone
    common_names = list(
        set(
            (gbif.get("common_names") or [])
            + ([inat.get("common_name")] if inat.get("common_name") else [])
        )
    )

    # Merge geographic info
    states = list(
        set(
            [normalize_location(s) for s in gbif.get("states_found", [])]
            + []  # iNaturalist doesn't provide state-level in our collection
        )
    )

    # Get the richest description from Wikipedia
    sections = wiki.get("sections", {}) if wiki else {}

    # Determine conservation status (prefer GBIF/iNaturalist, fallback to Wikipedia)
    conservation = ""
    if gbif.get("iucn_category"):
        conservation = normalize_conservation_status(gbif["iucn_category"])
    elif inat.get("conservation_status"):
        conservation = normalize_conservation_status(inat["conservation_status"])

    # Determine taxonomic group
    animal_class = gbif.get("class", "")
    if animal_class == "Aves":
        group = "birds"
    elif animal_class == "Mammalia":
        group = "mammals"
    elif animal_class == "Reptilia":
        group = "reptiles"
    else:
        group = "unknown"

    return {
        "species_id": gbif.get("species_key") or inat.get("taxon_id"),
        "scientific_name": gbif.get("canonical_name", "") or inat.get("scientific_name", ""),
        "common_names": common_names,
        "primary_name": common_names[0] if common_names else gbif.get("canonical_name", ""),
        "taxonomic_group": group,
        "taxonomy": {
            "kingdom": gbif.get("kingdom", "Animalia"),
            "phylum": gbif.get("phylum", "Chordata"),
            "class": gbif.get("class", ""),
            "order": gbif.get("order", ""),
            "family": gbif.get("family", ""),
            "genus": gbif.get("genus", ""),
        },
        "conservation_status": conservation,
        "geographic_regions": states,
        "observation_count": inat.get("observation_count", 0) or gbif.get("occurrences_india", 0),
        "description": {
            "introduction": sections.get("introduction", ""),
            "physical_description": sections.get("description", "")
            or sections.get("physical description", "")
            or sections.get("appearance", ""),
            "habitat": sections.get("habitat", "")
            or sections.get("habitat and ecology", "")
            or sections.get("distribution and habitat", "")
            or sections.get("habitat and distribution", ""),
            "behavior": sections.get("behaviour", "")
            or sections.get("behavior", "")
            or sections.get("ecology and behaviour", ""),
            "diet": sections.get("diet", "") or sections.get("feeding", ""),
            "reproduction": sections.get("reproduction", "") or sections.get("breeding", ""),
            "conservation": sections.get("conservation", "")
            or sections.get("conservation status", ""),
        },
        "sources": {
            "gbif_url": gbif.get("source_url", "") if gbif else "",
            "wikipedia_url": wiki.get("source_url", "") if wiki else "",
            "inaturalist_url": inat.get("source_url", "") if inat else "",
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    gbif_data = load_gbif_data()
    wiki_data = load_wikipedia_data()
    inat_data = load_inaturalist_data()

    # Collect all unique species keys
    all_keys = set(gbif_data.keys()) | set(wiki_data.keys()) | set(inat_data.keys())
    logger.info(f"Total unique species keys across all sources: {len(all_keys)}")

    # Merge
    merged = []
    for key in all_keys:
        gbif = gbif_data.get(key, {})
        wiki = wiki_data.get(key, {})
        inat = inat_data.get(key, {})

        # Skip if we have no substantial data
        if not gbif and not inat:
            continue

        record = merge_species(gbif, wiki, inat)

        # Skip if no meaningful description content
        desc = record["description"]
        has_content = any(
            [
                desc.get("introduction"),
                desc.get("physical_description"),
                desc.get("habitat"),
            ]
        )
        if not has_content and not record.get("common_names"):
            continue

        merged.append(record)

    # Sort by observation count (most observed first)
    merged.sort(key=lambda x: x.get("observation_count", 0), reverse=True)

    # Save merged data by group
    by_group = defaultdict(list)
    for sp in merged:
        by_group[sp["taxonomic_group"]].append(sp)

    for group, species_list in by_group.items():
        output_file = OUTPUT_DIR / f"{group}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(species_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(species_list)} {group} to {output_file}")

    # Save combined file
    with open(OUTPUT_DIR / "all_species.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Summary stats
    stats = {
        "total_species": len(merged),
        "by_group": {g: len(s) for g, s in by_group.items()},
        "with_description": sum(1 for s in merged if s["description"]["introduction"]),
        "with_habitat": sum(1 for s in merged if s["description"]["habitat"]),
        "with_conservation": sum(1 for s in merged if s["conservation_status"]),
        "with_regions": sum(1 for s in merged if s["geographic_regions"]),
    }
    with open(OUTPUT_DIR / "cleaning_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Cleaning complete. Total merged species: {len(merged)}")
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
