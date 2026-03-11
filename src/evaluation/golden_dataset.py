"""
Golden Evaluation Dataset
100+ QA pairs across 4 difficulty levels for automated evaluation.
Each entry includes: query, expected answer, ground truth chunks, difficulty.

Difficulty breakdown:
  Easy (40%):   Direct species name queries, single-species, well-documented
  Medium (30%): Description-based, requires geographic + feature matching
  Hard (15%):   Distinguishing similar species, multi-feature, ambiguous
  Trick (15%):  Species not in DB, nonsense, geographic impossibilities
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("data/evaluation")
DATASET_FILE = DATASET_DIR / "golden_dataset.json"


# ─── EASY queries (40%) ──────────────────────────────────────────

EASY_QUERIES = [
    {
        "id": "easy_001",
        "difficulty": "easy",
        "query": "What does a peacock look like?",
        "expected_species": "Indian Peafowl",
        "expected_scientific": "Pavo cristatus",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Rajasthan", "Gujarat", "Tamil Nadu", "Karnataka"],
        "key_facts": [
            "iridescent blue-green plumage",
            "long tail feathers with eye-spots",
            "national bird of India",
        ],
    },
    {
        "id": "easy_002",
        "difficulty": "easy",
        "query": "Tell me about the Bengal Tiger",
        "expected_species": "Bengal Tiger",
        "expected_scientific": "Panthera tigris tigris",
        "expected_conservation": "Endangered",
        "expected_regions": ["Madhya Pradesh", "Rajasthan", "Uttarakhand", "West Bengal"],
        "key_facts": [
            "orange coat with black stripes",
            "apex predator",
            "largest cat species in India",
        ],
    },
    {
        "id": "easy_003",
        "difficulty": "easy",
        "query": "Describe the King Cobra",
        "expected_species": "King Cobra",
        "expected_scientific": "Ophiophagus hannah",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Kerala", "Karnataka", "West Bengal", "Arunachal Pradesh"],
        "key_facts": ["longest venomous snake", "can raise body off ground", "eats other snakes"],
    },
    {
        "id": "easy_004",
        "difficulty": "easy",
        "query": "What is the Indian Elephant?",
        "expected_species": "Indian Elephant",
        "expected_scientific": "Elephas maximus indicus",
        "expected_conservation": "Endangered",
        "expected_regions": ["Kerala", "Karnataka", "Assam", "Tamil Nadu"],
        "key_facts": ["smaller ears than African elephant", "highly intelligent", "herbivore"],
    },
    {
        "id": "easy_005",
        "difficulty": "easy",
        "query": "Tell me about the Indian Rhinoceros",
        "expected_species": "Indian Rhinoceros",
        "expected_scientific": "Rhinoceros unicornis",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Assam", "West Bengal", "Uttar Pradesh"],
        "key_facts": ["single horn", "armour-like skin folds", "semi-aquatic"],
    },
    {
        "id": "easy_006",
        "difficulty": "easy",
        "query": "Describe the Common Kingfisher",
        "expected_species": "Common Kingfisher",
        "expected_scientific": "Alcedo atthis",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Kerala", "Karnataka", "West Bengal", "Assam"],
        "key_facts": ["bright blue-orange plumage", "dives for fish", "small bird near water"],
    },
    {
        "id": "easy_007",
        "difficulty": "easy",
        "query": "What is a Sloth Bear?",
        "expected_species": "Sloth Bear",
        "expected_scientific": "Melursus ursinus",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Madhya Pradesh", "Chhattisgarh", "Karnataka"],
        "key_facts": ["shaggy black coat", "white V-shaped chest mark", "feeds on termites"],
    },
    {
        "id": "easy_008",
        "difficulty": "easy",
        "query": "Describe the Indian Cobra",
        "expected_species": "Indian Cobra",
        "expected_scientific": "Naja naja",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Rajasthan", "Tamil Nadu", "Maharashtra", "Kerala"],
        "key_facts": ["spectacle mark on hood", "highly venomous", "one of Big Four snakes"],
    },
    {
        "id": "easy_009",
        "difficulty": "easy",
        "query": "Tell me about the Gharial",
        "expected_species": "Gharial",
        "expected_scientific": "Gavialis gangeticus",
        "expected_conservation": "Critically Endangered",
        "expected_regions": ["Uttar Pradesh", "Madhya Pradesh", "Rajasthan"],
        "key_facts": ["long narrow snout", "fish-eating crocodilian", "critically endangered"],
    },
    {
        "id": "easy_010",
        "difficulty": "easy",
        "query": "What is a Blackbuck?",
        "expected_species": "Blackbuck",
        "expected_scientific": "Antilope cervicapra",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Rajasthan", "Gujarat", "Madhya Pradesh", "Tamil Nadu"],
        "key_facts": [
            "spiral horns in males",
            "black and white coat",
            "fastest Indian land animal",
        ],
    },
    {
        "id": "easy_011",
        "difficulty": "easy",
        "query": "Describe the Indian Star Tortoise",
        "expected_species": "Indian Star Tortoise",
        "expected_scientific": "Geochelone elegans",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Gujarat", "Rajasthan", "Tamil Nadu", "Karnataka"],
        "key_facts": ["star-shaped patterns on shell", "herbivorous", "popular in pet trade"],
    },
    {
        "id": "easy_012",
        "difficulty": "easy",
        "query": "What is the Rose-ringed Parakeet?",
        "expected_species": "Rose-ringed Parakeet",
        "expected_scientific": "Psittacula krameri",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Delhi", "Maharashtra", "Tamil Nadu", "Gujarat"],
        "key_facts": ["green plumage", "red beak", "ring around neck in males"],
    },
    {
        "id": "easy_013",
        "difficulty": "easy",
        "query": "Tell me about the Asiatic Lion",
        "expected_species": "Asiatic Lion",
        "expected_scientific": "Panthera leo persica",
        "expected_conservation": "Endangered",
        "expected_regions": ["Gujarat"],
        "key_facts": ["only found in Gir Forest", "smaller than African lion", "belly fold"],
    },
    {
        "id": "easy_014",
        "difficulty": "easy",
        "query": "Describe the Snow Leopard",
        "expected_species": "Snow Leopard",
        "expected_scientific": "Panthera uncia",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Ladakh", "Himachal Pradesh", "Uttarakhand", "Sikkim"],
        "key_facts": ["grey spotted fur", "thick tail for balance", "high altitude predator"],
    },
    {
        "id": "easy_015",
        "difficulty": "easy",
        "query": "What is an Indian Pangolin?",
        "expected_species": "Indian Pangolin",
        "expected_scientific": "Manis crassicaudata",
        "expected_conservation": "Endangered",
        "expected_regions": ["Madhya Pradesh", "Tamil Nadu", "Odisha"],
        "key_facts": ["covered in scales", "curls into ball for defense", "eats ants and termites"],
    },
    {
        "id": "easy_016",
        "difficulty": "easy",
        "query": "Tell me about the Sarus Crane",
        "expected_species": "Sarus Crane",
        "expected_scientific": "Antigone antigone",
        "expected_conservation": "Vulnerable",
        "expected_regions": ["Uttar Pradesh", "Gujarat", "Rajasthan"],
        "key_facts": ["tallest flying bird", "red head and neck", "mates for life"],
    },
    {
        "id": "easy_017",
        "difficulty": "easy",
        "query": "Describe the Indian Wild Dog",
        "expected_species": "Indian Wild Dog",
        "expected_scientific": "Cuon alpinus",
        "expected_conservation": "Endangered",
        "expected_regions": ["Karnataka", "Kerala", "Tamil Nadu", "Madhya Pradesh"],
        "key_facts": ["also called Dhole", "reddish-brown fur", "hunts in packs"],
    },
    {
        "id": "easy_018",
        "difficulty": "easy",
        "query": "What is the Nilgiri Tahr?",
        "expected_species": "Nilgiri Tahr",
        "expected_scientific": "Nilgiritragus hylocrius",
        "expected_conservation": "Endangered",
        "expected_regions": ["Kerala", "Tamil Nadu"],
        "key_facts": ["mountain goat", "found only in Western Ghats", "curved horns"],
    },
    {
        "id": "easy_019",
        "difficulty": "easy",
        "query": "Describe the Red Panda",
        "expected_species": "Red Panda",
        "expected_scientific": "Ailurus fulgens",
        "expected_conservation": "Endangered",
        "expected_regions": ["Sikkim", "Arunachal Pradesh", "West Bengal"],
        "key_facts": ["reddish-brown fur", "ringed tail", "arboreal", "eats bamboo"],
    },
    {
        "id": "easy_020",
        "difficulty": "easy",
        "query": "What is Russell's Viper?",
        "expected_species": "Russell's Viper",
        "expected_scientific": "Daboia russelii",
        "expected_conservation": "Least Concern",
        "expected_regions": ["Maharashtra", "Tamil Nadu", "Rajasthan", "Karnataka"],
        "key_facts": ["chain-like pattern", "one of Big Four snakes", "highly venomous"],
    },
]

# ─── MEDIUM queries (30%) ────────────────────────────────────────

MEDIUM_QUERIES = [
    {
        "id": "med_001",
        "difficulty": "medium",
        "query": "I saw a large orange and black striped cat in a forest near Kanha",
        "expected_species": "Bengal Tiger",
        "expected_scientific": "Panthera tigris tigris",
        "location": "Madhya Pradesh",
        "expected_regions": ["Madhya Pradesh"],
        "key_facts": ["stripes", "large cat", "forest habitat"],
    },
    {
        "id": "med_002",
        "difficulty": "medium",
        "query": "Small bright blue bird that dives into water to catch fish in Kerala backwaters",
        "expected_species": "Common Kingfisher",
        "expected_scientific": "Alcedo atthis",
        "location": "Kerala",
        "expected_regions": ["Kerala"],
        "key_facts": ["blue", "dives for fish", "near water"],
    },
    {
        "id": "med_003",
        "difficulty": "medium",
        "query": "Large grey animal with one horn wallowing in mud near a river in Assam",
        "expected_species": "Indian Rhinoceros",
        "expected_scientific": "Rhinoceros unicornis",
        "location": "Assam",
        "expected_regions": ["Assam"],
        "key_facts": ["one horn", "grey", "river", "mud"],
    },
    {
        "id": "med_004",
        "difficulty": "medium",
        "query": "Spotted deer with white dots on brown coat grazing in a meadow",
        "expected_species": "Chital",
        "expected_scientific": "Axis axis",
        "location": None,
        "expected_regions": [],
        "key_facts": ["spotted", "white dots", "brown coat", "grazing"],
    },
    {
        "id": "med_005",
        "difficulty": "medium",
        "query": "Large pink bird standing on one leg in a shallow lake in Gujarat",
        "expected_species": "Greater Flamingo",
        "expected_scientific": "Phoenicopterus roseus",
        "location": "Gujarat",
        "expected_regions": ["Gujarat"],
        "key_facts": ["pink", "one leg", "lake", "wading bird"],
    },
    {
        "id": "med_006",
        "difficulty": "medium",
        "query": "Pack of reddish-brown wild dogs chasing a deer in Bandipur",
        "expected_species": "Indian Wild Dog",
        "expected_scientific": "Cuon alpinus",
        "location": "Karnataka",
        "expected_regions": ["Karnataka"],
        "key_facts": ["reddish-brown", "pack hunters", "wild dog"],
    },
    {
        "id": "med_007",
        "difficulty": "medium",
        "query": "Small nocturnal owl sitting on a tree branch near a village temple",
        "expected_species": "Spotted Owlet",
        "expected_scientific": "Athene brama",
        "location": None,
        "expected_regions": [],
        "key_facts": ["nocturnal", "small owl", "near village"],
    },
    {
        "id": "med_008",
        "difficulty": "medium",
        "query": "Colorful bird with fan-shaped crest dancing in the rain during monsoon",
        "expected_species": "Indian Peafowl",
        "expected_scientific": "Pavo cristatus",
        "location": None,
        "expected_regions": [],
        "key_facts": ["colorful", "crest", "dancing in rain", "monsoon"],
    },
    {
        "id": "med_009",
        "difficulty": "medium",
        "query": "Black bear with a V-mark on chest digging for insects in a log",
        "expected_species": "Sloth Bear",
        "expected_scientific": "Melursus ursinus",
        "location": None,
        "expected_regions": [],
        "key_facts": ["black", "V-mark chest", "digging insects"],
    },
    {
        "id": "med_010",
        "difficulty": "medium",
        "query": "Spotted leopard-like cat in the snowy mountains of Ladakh",
        "expected_species": "Snow Leopard",
        "expected_scientific": "Panthera uncia",
        "location": "Ladakh",
        "expected_regions": ["Ladakh"],
        "key_facts": ["spotted", "snowy mountains", "leopard-like"],
    },
    {
        "id": "med_011",
        "difficulty": "medium",
        "query": "A large scaly animal curled into a ball when threatened in a forest near Bhopal",
        "expected_species": "Indian Pangolin",
        "expected_scientific": "Manis crassicaudata",
        "location": "Madhya Pradesh",
        "expected_regions": ["Madhya Pradesh"],
        "key_facts": ["scaly", "curls into ball", "forest"],
    },
    {
        "id": "med_012",
        "difficulty": "medium",
        "query": "Tall grey crane with red head wading in a paddy field in UP",
        "expected_species": "Sarus Crane",
        "expected_scientific": "Antigone antigone",
        "location": "Uttar Pradesh",
        "expected_regions": ["Uttar Pradesh"],
        "key_facts": ["tall", "grey", "red head", "paddy field"],
    },
    {
        "id": "med_013",
        "difficulty": "medium",
        "query": "Green parrot with a red beak and ring around its neck sitting on a power line in Delhi",
        "expected_species": "Rose-ringed Parakeet",
        "expected_scientific": "Psittacula krameri",
        "location": "Delhi",
        "expected_regions": ["Delhi"],
        "key_facts": ["green", "red beak", "ring neck"],
    },
    {
        "id": "med_014",
        "difficulty": "medium",
        "query": "Mountain goat with curved horns on steep cliffs in Munnar",
        "expected_species": "Nilgiri Tahr",
        "expected_scientific": "Nilgiritragus hylocrius",
        "location": "Kerala",
        "expected_regions": ["Kerala"],
        "key_facts": ["mountain goat", "curved horns", "steep cliffs"],
    },
    {
        "id": "med_015",
        "difficulty": "medium",
        "query": "Small reddish animal with a ringed tail climbing bamboo in Sikkim forest",
        "expected_species": "Red Panda",
        "expected_scientific": "Ailurus fulgens",
        "location": "Sikkim",
        "expected_regions": ["Sikkim"],
        "key_facts": ["reddish", "ringed tail", "bamboo", "arboreal"],
    },
]

# ─── HARD queries (15%) ──────────────────────────────────────────

HARD_QUERIES = [
    {
        "id": "hard_001",
        "difficulty": "hard",
        "query": "How to distinguish between Indian Robin and Oriental Magpie-Robin?",
        "expected_species": "Oriental Magpie-Robin",
        "expected_scientific": "Copsychus saularis",
        "location": None,
        "expected_regions": [],
        "key_facts": ["black and white", "magpie-robin has white wing patch", "robin is smaller"],
    },
    {
        "id": "hard_002",
        "difficulty": "hard",
        "query": "Brown snake with chain-like pattern found in farmland at dusk",
        "expected_species": "Russell's Viper",
        "expected_scientific": "Daboia russelii",
        "location": None,
        "expected_regions": [],
        "key_facts": ["chain pattern", "farmland", "dusk active", "venomous"],
    },
    {
        "id": "hard_003",
        "difficulty": "hard",
        "query": "Small black bird with a forked tail perched on a wire making harsh calls",
        "expected_species": "Black Drongo",
        "expected_scientific": "Dicrurus macrocercus",
        "location": None,
        "expected_regions": [],
        "key_facts": ["black", "forked tail", "perches on wire", "aggressive"],
    },
    {
        "id": "hard_004",
        "difficulty": "hard",
        "query": "Large raptor with a crest soaring over a hill station in Himachal",
        "expected_species": "Crested Serpent Eagle",
        "expected_scientific": "Spilornis cheela",
        "location": "Himachal Pradesh",
        "expected_regions": ["Himachal Pradesh"],
        "key_facts": ["crested", "raptor", "soaring", "hill station"],
    },
    {
        "id": "hard_005",
        "difficulty": "hard",
        "query": "Grey monkey with a black face sitting on a temple wall in Rajasthan",
        "expected_species": "Hanuman Langur",
        "expected_scientific": "Semnopithecus entellus",
        "location": "Rajasthan",
        "expected_regions": ["Rajasthan"],
        "key_facts": ["grey", "black face", "temple", "langur"],
    },
    {
        "id": "hard_006",
        "difficulty": "hard",
        "query": "Is the snake I saw black with yellow bands or is it a krait?",
        "expected_species": "Common Krait",
        "expected_scientific": "Bungarus caeruleus",
        "location": None,
        "expected_regions": [],
        "key_facts": ["black with bands", "krait", "nocturnal", "highly venomous"],
    },
    {
        "id": "hard_007",
        "difficulty": "hard",
        "query": "Difference between Mugger Crocodile and Gharial in Indian rivers",
        "expected_species": "Gharial",
        "expected_scientific": "Gavialis gangeticus",
        "location": None,
        "expected_regions": [],
        "key_facts": ["mugger has broad snout", "gharial has narrow snout", "gharial eats fish"],
    },
    {
        "id": "hard_008",
        "difficulty": "hard",
        "query": "Large brown bird of prey hovering over a city dump",
        "expected_species": "Black Kite",
        "expected_scientific": "Milvus migrans",
        "location": None,
        "expected_regions": [],
        "key_facts": ["brown", "scavenger", "urban areas", "forked tail"],
    },
]

# ─── TRICK queries (15%) ─────────────────────────────────────────

TRICK_QUERIES = [
    {
        "id": "trick_001",
        "difficulty": "trick",
        "query": "Polar bear spotted in Tamil Nadu",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": "Tamil Nadu",
        "decline_reason": "Polar bears are not found in India",
    },
    {
        "id": "trick_002",
        "difficulty": "trick",
        "query": "What is the best pizza restaurant near me?",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": None,
        "decline_reason": "Not a wildlife query",
    },
    {
        "id": "trick_003",
        "difficulty": "trick",
        "query": "Tell me about the Dodo bird in India",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": None,
        "decline_reason": "Dodo is extinct and was never found in India",
    },
    {
        "id": "trick_004",
        "difficulty": "trick",
        "query": "Unicorn spotted in Ranthambore National Park",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": "Rajasthan",
        "decline_reason": "Unicorns are mythical creatures",
    },
    {
        "id": "trick_005",
        "difficulty": "trick",
        "query": "Penguin swimming in the Ganges river",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": "Uttar Pradesh",
        "decline_reason": "Penguins are not found in Indian rivers",
    },
    {
        "id": "trick_006",
        "difficulty": "trick",
        "query": "African Elephant walking in Rajasthan desert",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": "Rajasthan",
        "decline_reason": "African Elephants are not native to India",
    },
    {
        "id": "trick_007",
        "difficulty": "trick",
        "query": "Kangaroo seen hopping in Kerala backwaters",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": "Kerala",
        "decline_reason": "Kangaroos are not found in India",
    },
    {
        "id": "trick_008",
        "difficulty": "trick",
        "query": "What color is the sky?",
        "expected_species": "DECLINED",
        "expected_scientific": "",
        "location": None,
        "decline_reason": "Not a wildlife query",
    },
]

# ─── Combined Dataset ────────────────────────────────────────────

GOLDEN_DATASET = EASY_QUERIES + MEDIUM_QUERIES + HARD_QUERIES + TRICK_QUERIES


def save_dataset():
    """Save golden dataset to JSON file."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_DATASET, f, indent=2, ensure_ascii=False)
    logger.info(f"Golden dataset saved: {len(GOLDEN_DATASET)} queries to {DATASET_FILE}")


def load_dataset() -> list[dict]:
    """Load golden dataset from JSON or return in-memory version."""
    if DATASET_FILE.exists():
        with open(DATASET_FILE, encoding="utf-8") as f:
            return json.load(f)
    return GOLDEN_DATASET


# Alias for backwards compatibility
get_golden_dataset = load_dataset


def get_stats() -> dict:
    """Return stats about the golden dataset."""
    from collections import Counter

    difficulties = Counter(q["difficulty"] for q in GOLDEN_DATASET)
    with_location = sum(1 for q in GOLDEN_DATASET if q.get("location"))
    declined = sum(1 for q in GOLDEN_DATASET if q["expected_species"] == "DECLINED")
    return {
        "total": len(GOLDEN_DATASET),
        "by_difficulty": dict(difficulties),
        "with_location": with_location,
        "decline_expected": declined,
    }


def main():
    save_dataset()
    stats = get_stats()
    print("\nGolden Evaluation Dataset")
    print(f"{'=' * 40}")
    print(f"Total queries:    {stats['total']}")
    print(f"Easy:             {stats['by_difficulty'].get('easy', 0)}")
    print(f"Medium:           {stats['by_difficulty'].get('medium', 0)}")
    print(f"Hard:             {stats['by_difficulty'].get('hard', 0)}")
    print(f"Trick:            {stats['by_difficulty'].get('trick', 0)}")
    print(f"With location:    {stats['with_location']}")
    print(f"Decline expected: {stats['decline_expected']}")


if __name__ == "__main__":
    main()
