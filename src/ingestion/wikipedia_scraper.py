"""
Wikipedia Species Scraper
Fetches species articles from Wikipedia for detailed descriptions,
habitat info, behavior, and conservation status.
Uses the Wikipedia API (no scraping HTML).
"""

import json
import logging
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
OUTPUT_DIR = Path("data/raw/wikipedia")

# Wikipedia requires a User-Agent header identifying the bot
HEADERS = {
    "User-Agent": "WildlifeTrackerBot/1.0 (educational project; Python/requests)",
    "Accept": "application/json",
}

# Reusable session with headers pre-set
_session = requests.Session()
_session.headers.update(HEADERS)

# Sections we care about for wildlife
RELEVANT_SECTIONS = [
    "description",
    "physical description",
    "appearance",
    "morphology",
    "habitat",
    "habitat and ecology",
    "distribution",
    "range",
    "distribution and habitat",
    "habitat and distribution",
    "behaviour",
    "behavior",
    "diet",
    "feeding",
    "reproduction",
    "breeding",
    "conservation",
    "conservation status",
    "threats",
    "population",
    "ecology",
    "ecology and behaviour",
    "ecology and behavior",
    "taxonomy",
    "classification",
    "cultural significance",
    "in culture",
]


def search_wikipedia(query: str) -> str | None:
    """Search Wikipedia and return the best matching page title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
    }
    try:
        resp = _session.get(WIKI_API, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except requests.RequestException as e:
        logger.error(f"Wikipedia search failed for '{query}': {e}")
    return None


def get_page_sections(title: str) -> list[dict]:
    """Get the section structure of a Wikipedia page."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
    }
    try:
        resp = _session.get(WIKI_API, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("parse", {}).get("sections", [])
    except requests.RequestException:
        return []


def get_page_extract(title: str) -> str:
    """Get the plain text extract (intro) of a Wikipedia page."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    try:
        resp = _session.get(WIKI_API, params=params, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except requests.RequestException:
        pass
    return ""


def get_section_text(title: str, section_index: int) -> str:
    """Get plain text content of a specific section."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": section_index,
        "format": "json",
    }
    try:
        resp = _session.get(WIKI_API, params=params, timeout=30)
        resp.raise_for_status()
        wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
        return clean_wikitext(wikitext)
    except requests.RequestException:
        return ""


def clean_wikitext(text: str) -> str:
    """Clean wikitext markup to plain text."""
    # Remove references
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/>", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove templates (simple ones)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove file/image links
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Convert wiki links [[target|display]] to display
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    # Convert simple wiki links [[target]] to target
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)
    # Remove section headers from within text
    text = re.sub(r"={2,5}[^=]+=+", "", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def scrape_species_page(species_name: str, scientific_name: str = "") -> dict | None:
    """Scrape a full species article from Wikipedia."""
    # Try scientific name first, then common name
    title = None
    for query in [
        scientific_name,
        species_name,
        f"{species_name} (bird)",
        f"{species_name} (animal)",
    ]:
        if not query:
            continue
        title = search_wikipedia(query)
        if title:
            break

    if not title:
        logger.warning(f"No Wikipedia article found for: {species_name}")
        return None

    # Get intro
    intro = get_page_extract(title)
    if not intro or len(intro) < 50:
        logger.warning(f"Article too short for: {title}")
        return None

    # Get sections
    sections_meta = get_page_sections(title)
    sections = {"introduction": intro}

    for sec in sections_meta:
        sec_name = sec.get("line", "").lower().strip()
        sec_index = sec.get("index")

        if any(relevant in sec_name for relevant in RELEVANT_SECTIONS):
            text = get_section_text(title, sec_index)
            if text and len(text) > 30:
                sections[sec_name] = text
            time.sleep(0.2)

    return {
        "species_name": species_name,
        "scientific_name": scientific_name,
        "wikipedia_title": title,
        "sections": sections,
        "source": "wikipedia",
        "source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }


def load_gbif_species() -> list[dict]:
    """Load species names from GBIF collected data."""
    gbif_dir = Path("data/raw/gbif")
    species = []

    for filepath in gbif_dir.glob("*.json"):
        if filepath.name == "collection_summary.json":
            continue
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for sp in data:
                names = sp.get("common_names", [])
                common_name = names[0] if names else sp.get("canonical_name", "")
                species.append(
                    {
                        "common_name": common_name,
                        "scientific_name": sp.get("canonical_name", ""),
                        "group": filepath.stem,
                    }
                )

    return species


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load species from GBIF data
    species_list = load_gbif_species()

    if not species_list:
        logger.warning("No GBIF data found. Using sample species from config.")
        import yaml

        config_path = Path("config/species_list.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        species_list = []
        for group_name, group_data in config.get("taxonomic_groups", {}).items():
            for sp_name in group_data.get("sample_species", []):
                species_list.append(
                    {
                        "common_name": sp_name,
                        "scientific_name": "",
                        "group": group_name,
                    }
                )

    logger.info(f"Scraping Wikipedia for {len(species_list)} species...")

    results = {"birds": [], "mammals": [], "reptiles": []}
    failed = []

    for sp in tqdm(species_list, desc="Scraping Wikipedia"):
        article = scrape_species_page(sp["common_name"], sp.get("scientific_name", ""))
        if article:
            group = sp.get("group", "unknown")
            if group in results:
                results[group].append(article)
        else:
            failed.append(sp["common_name"])
        time.sleep(0.5)  # Rate limiting

    # Save results
    for group_name, articles in results.items():
        if articles:
            output_file = OUTPUT_DIR / f"{group_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(articles)} {group_name} articles to {output_file}")

    # Save failures for review
    if failed:
        with open(OUTPUT_DIR / "failed_species.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        logger.warning(f"{len(failed)} species failed. Saved to failed_species.json")

    total = sum(len(v) for v in results.values())
    logger.info(f"Wikipedia scraping complete. Total articles: {total}")


if __name__ == "__main__":
    main()
