"""
GBIF Data Collector
Downloads species occurrence records for Indian wildlife from GBIF API.
Focuses on birds, mammals, and reptiles found in India.
"""

import json
import logging
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GBIF_API = "https://api.gbif.org/v1"
INDIA_COUNTRY_CODE = "IN"
OUTPUT_DIR = Path("data/raw/gbif")

# GBIF taxon keys for major groups
TAXON_KEYS = {
    "birds": 212,  # Class Aves
    "mammals": 359,  # Class Mammalia
    "reptiles": 358,  # Class Reptilia
}


def search_species(class_key: int, limit: int = 500, offset: int = 0) -> list[dict]:
    """Search GBIF for species in a taxonomic class found in India."""
    url = f"{GBIF_API}/species/search"
    params = {
        "highertaxonKey": class_key,
        "country": INDIA_COUNTRY_CODE,
        "rank": "SPECIES",
        "status": "ACCEPTED",
        "limit": limit,
        "offset": offset,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException as e:
        logger.error(f"GBIF species search failed: {e}")
        return []


def get_species_details(species_key: int) -> dict | None:
    """Get detailed info for a specific species from GBIF."""
    url = f"{GBIF_API}/species/{species_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to get species {species_key}: {e}")
        return None


def get_species_vernacular_names(species_key: int) -> list[str]:
    """Get common/vernacular names for a species."""
    url = f"{GBIF_API}/species/{species_key}/vernacularNames"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        names = []
        for r in results:
            if r.get("language", "") in ("eng", "en", ""):
                name = r.get("vernacularName", "")
                if name and name not in names:
                    names.append(name)
        return names[:5]
    except requests.RequestException:
        return []


def get_occurrence_summary(species_key: int) -> dict:
    """Get occurrence count and sample locations in India for a species."""
    url = f"{GBIF_API}/occurrence/search"
    params = {
        "taxonKey": species_key,
        "country": INDIA_COUNTRY_CODE,
        "limit": 20,
        "hasCoordinate": True,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        locations = []
        for occ in data.get("results", []):
            loc = {
                "state": occ.get("stateProvince", ""),
                "lat": occ.get("decimalLatitude"),
                "lon": occ.get("decimalLongitude"),
            }
            if loc["state"] and loc not in locations:
                locations.append(loc)

        return {
            "total_occurrences_india": data.get("count", 0),
            "sample_locations": locations[:10],
            "states_found": list(set(loc["state"] for loc in locations if loc["state"])),
        }
    except requests.RequestException:
        return {"total_occurrences_india": 0, "sample_locations": [], "states_found": []}


def collect_species_for_group(
    group_name: str, class_key: int, max_species: int = 500
) -> list[dict]:
    """Collect all species data for a taxonomic group."""
    logger.info(f"Collecting {group_name} species from GBIF...")
    all_species = []
    offset = 0
    batch_size = 100

    while offset < max_species:
        results = search_species(class_key, limit=batch_size, offset=offset)
        if not results:
            break
        all_species.extend(results)
        offset += batch_size
        time.sleep(0.5)  # Rate limiting

    logger.info(f"Found {len(all_species)} {group_name} species. Enriching with details...")

    enriched = []
    for sp in tqdm(all_species, desc=f"Enriching {group_name}"):
        species_key = sp.get("key")
        if not species_key:
            continue

        vernacular = get_species_vernacular_names(species_key)
        occurrences = get_occurrence_summary(species_key)
        time.sleep(0.3)  # Rate limiting

        record = {
            "species_key": species_key,
            "scientific_name": sp.get("scientificName", ""),
            "canonical_name": sp.get("canonicalName", ""),
            "common_names": vernacular,
            "kingdom": sp.get("kingdom", ""),
            "phylum": sp.get("phylum", ""),
            "class": sp.get("class", ""),
            "order": sp.get("order", ""),
            "family": sp.get("family", ""),
            "genus": sp.get("genus", ""),
            "taxonomic_status": sp.get("taxonomicStatus", ""),
            "iucn_category": sp.get("iucnRedListCategory", ""),
            "occurrences_india": occurrences["total_occurrences_india"],
            "states_found": occurrences["states_found"],
            "sample_locations": occurrences["sample_locations"],
            "source": "gbif",
            "source_url": f"https://www.gbif.org/species/{species_key}",
        }
        enriched.append(record)

    return enriched


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config for target counts
    config_path = Path("config/species_list.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        groups = config.get("taxonomic_groups", {})
    else:
        groups = {
            "birds": {"target_count": 400},
            "mammals": {"target_count": 150},
            "reptiles": {"target_count": 100},
        }

    all_data = {}
    total = 0

    for group_name, class_key in TAXON_KEYS.items():
        max_sp = groups.get(group_name, {}).get("target_count", 200)
        species_list = collect_species_for_group(group_name, class_key, max_species=max_sp)

        output_file = OUTPUT_DIR / f"{group_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(species_list, f, indent=2, ensure_ascii=False)

        all_data[group_name] = len(species_list)
        total += len(species_list)
        logger.info(f"Saved {len(species_list)} {group_name} to {output_file}")

    # Summary
    summary = {
        "source": "GBIF",
        "country": "India",
        "collection_date": time.strftime("%Y-%m-%d"),
        "counts": all_data,
        "total_species": total,
    }
    with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"GBIF collection complete. Total species: {total}")


if __name__ == "__main__":
    main()
