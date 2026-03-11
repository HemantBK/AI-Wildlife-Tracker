# Data Sources

## Sources

### 1. GBIF (Global Biodiversity Information Facility)
- **URL**: https://www.gbif.org
- **Data**: Species occurrence records for India (birds, mammals, reptiles)
- **License**: CC BY 4.0 — free to use with attribution
- **Script**: `src/ingestion/gbif_collector.py`

### 2. iNaturalist
- **URL**: https://www.inaturalist.org
- **Data**: Community observation data, species descriptions, photos metadata
- **License**: Mixed (CC BY-NC per observation) — non-commercial use
- **Script**: `src/ingestion/inaturalist_collector.py`

### 3. Wikipedia
- **URL**: https://en.wikipedia.org
- **Data**: Species descriptions, habitat, behavior, conservation status
- **License**: CC BY-SA 3.0 — share-alike required
- **Script**: `src/ingestion/wikipedia_scraper.py`

## Directory Structure

```
data/
├── raw/          # Original downloaded data (not committed to git)
├── processed/    # Cleaned, normalized, deduplicated species records
└── chunks/       # Final chunked data with metadata, ready for embedding
```

## Reproducing the Data Pipeline

```bash
# Download raw data
python -m src.ingestion.gbif_collector
python -m src.ingestion.wikipedia_scraper
python -m src.ingestion.inaturalist_collector

# Clean and chunk
python -m src.preprocessing.cleaner
python -m src.preprocessing.chunker

# Validate
python -m src.preprocessing.validator

# Or run everything at once:
make data-pipeline
```

## Last Updated
- Initial data collection: Phase 1
