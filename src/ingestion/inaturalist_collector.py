"""
iNaturalist Data Collector
Fetches species observation data from iNaturalist API for Indian wildlife.
Supplements GBIF data with community observations and descriptions.
"""

import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INAT_API = "https://api.inaturalist.org/v1"
OUTPUT_DIR = Path("data/raw/inaturalist")

# India bounding box (approximate)
INDIA_BBOX = {
    "nelat": 35.5,  # North
    "nelng": 97.4,  # East
    "swlat": 6.7,  # South
    "swlng": 68.1,  # West
}

# Iconic taxa IDs in iNaturalist
ICONIC_TAXA = {
    "birds": "Aves",
    "mammals": "Mammalia",
    "reptiles": "Reptilia",
}


def search_species(taxon_name: str, per_page: int = 200, page: int = 1) -> list[dict]:
    """Search iNaturalist for observed species of a taxonomic group in India."""
    url = f"{INAT_API}/observations/species_counts"
    params = {
        "iconic_taxa": taxon_name,
        "place_id": 6681,  # India place ID in iNaturalist
        "quality_grade": "research",
        "per_page": per_page,
        "page": page,
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException as e:
        logger.error(f"iNaturalist species search failed: {e}")
        return []


def get_taxon_details(taxon_id: int) -> dict | None:
    """Get detailed taxon information from iNaturalist."""
    url = f"{INAT_API}/taxa/{taxon_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except requests.RequestException as e:
        logger.error(f"Failed to get taxon {taxon_id}: {e}")
        return None


def get_observation_stats(taxon_id: int) -> dict:
    """Get observation statistics for a species in India."""
    url = f"{INAT_API}/observations"
    params = {
        "taxon_id": taxon_id,
        "place_id": 6681,
        "quality_grade": "research",
        "per_page": 0,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return {"observation_count": resp.json().get("total_results", 0)}
    except requests.RequestException:
        return {"observation_count": 0}


def collect_group(group_name: str, taxon_name: str, max_species: int = 500) -> list[dict]:
    """Collect species data for a taxonomic group."""
    logger.info(f"Collecting {group_name} from iNaturalist...")
    all_results = []
    page = 1

    while len(all_results) < max_species:
        results = search_species(taxon_name, per_page=200, page=page)
        if not results:
            break
        all_results.extend(results)
        page += 1
        time.sleep(1)  # Rate limiting

    all_results = all_results[:max_species]
    logger.info(f"Found {len(all_results)} {group_name}. Enriching...")

    enriched = []
    for item in tqdm(all_results, desc=f"Enriching {group_name}"):
        taxon = item.get("taxon", {})
        taxon_id = taxon.get("id")
        if not taxon_id:
            continue

        # Get extra details
        details = get_taxon_details(taxon_id)
        time.sleep(0.3)

        wikipedia_summary = ""
        conservation_status = ""
        if details:
            wikipedia_summary = details.get("wikipedia_summary", "") or ""
            cs = details.get("conservation_status")
            if cs:
                conservation_status = cs.get("status", "")

        record = {
            "taxon_id": taxon_id,
            "scientific_name": taxon.get("name", ""),
            "common_name": taxon.get("preferred_common_name", ""),
            "iconic_taxon": taxon.get("iconic_taxon_name", ""),
            "rank": taxon.get("rank", ""),
            "ancestry": taxon.get("ancestry", ""),
            "observation_count": item.get("count", 0),
            "conservation_status": conservation_status,
            "wikipedia_summary": wikipedia_summary,
            "photo_url": taxon.get("default_photo", {}).get("medium_url", "")
            if taxon.get("default_photo")
            else "",
            "source": "inaturalist",
            "source_url": f"https://www.inaturalist.org/taxa/{taxon_id}",
        }
        enriched.append(record)

    return enriched


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_counts = {"birds": 400, "mammals": 150, "reptiles": 100}
    all_data = {}
    total = 0

    for group_name, taxon_name in ICONIC_TAXA.items():
        max_sp = target_counts.get(group_name, 200)
        species_list = collect_group(group_name, taxon_name, max_species=max_sp)

        output_file = OUTPUT_DIR / f"{group_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(species_list, f, indent=2, ensure_ascii=False)

        all_data[group_name] = len(species_list)
        total += len(species_list)
        logger.info(f"Saved {len(species_list)} {group_name} to {output_file}")

    summary = {
        "source": "iNaturalist",
        "region": "India",
        "collection_date": time.strftime("%Y-%m-%d"),
        "counts": all_data,
        "total_species": total,
    }
    with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"iNaturalist collection complete. Total species: {total}")


if __name__ == "__main__":
    main()
