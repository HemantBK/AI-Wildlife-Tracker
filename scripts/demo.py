"""
Demo Script
Runs a set of example queries through the API and displays results.
Useful for demos, screenshots, and verifying the system works end-to-end.

Usage:
  python scripts/demo.py              # Run against local API
  python scripts/demo.py --api-url http://localhost:8000
"""

import sys
import time
import json
import requests


API_BASE = "http://localhost:8000"

DEMO_QUERIES = [
    {
        "name": "Easy: Bengal Tiger",
        "query": "I saw a large orange and black striped cat in the forest near a river",
        "location": "Madhya Pradesh",
        "season": "winter",
    },
    {
        "name": "Easy: Indian Peafowl",
        "query": "A colorful bird with a fan-shaped crest that dances beautifully in the rain",
        "location": "Rajasthan",
    },
    {
        "name": "Medium: Indian Rhino",
        "query": "Large grey animal with one horn spotted near a river in a national park",
        "location": "Assam",
    },
    {
        "name": "Medium: Description-based",
        "query": "Small bright blue bird with a long tail sitting on a branch near a stream",
        "location": "Kerala",
        "season": "monsoon",
    },
    {
        "name": "Hard: Similar species",
        "query": "Spotted cat climbing a tree in the Western Ghats forest",
        "location": "Karnataka",
    },
    {
        "name": "Trick: Out of range",
        "query": "I think I saw a polar bear in Tamil Nadu near the beach",
        "location": "Tamil Nadu",
    },
    {
        "name": "Trick: Non-wildlife",
        "query": "Can you recommend the best pizza restaurants in Mumbai?",
        "location": "Maharashtra",
    },
]


def run_demo():
    print("\n" + "=" * 70)
    print("  Wildlife Tracker - Demo")
    print("=" * 70)

    # Parse args
    api_url = API_BASE
    for i, arg in enumerate(sys.argv):
        if arg == "--api-url" and i + 1 < len(sys.argv):
            api_url = sys.argv[i + 1]

    # Check API health
    print(f"\n  API: {api_url}")
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        print(f"  Status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"  ERROR: Cannot reach API at {api_url}")
        print(f"  Make sure the server is running: make api")
        return 1

    print(f"\n  Running {len(DEMO_QUERIES)} demo queries...\n")

    results = []
    for i, demo in enumerate(DEMO_QUERIES):
        name = demo.pop("name")
        print(f"{'─' * 70}")
        print(f"  [{i+1}/{len(DEMO_QUERIES)}] {name}")
        print(f"  Query: {demo['query'][:80]}...")
        if demo.get("location"):
            print(f"  Location: {demo.get('location')}")

        try:
            start = time.time()
            resp = requests.post(f"{api_url}/identify", json=demo, timeout=35)
            elapsed = time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                species = data.get("species_name", "N/A")
                confidence = data.get("confidence", 0)
                reasoning = data.get("reasoning", "")[:120]

                # Color-coded confidence
                if confidence >= 0.8:
                    conf_label = "HIGH"
                elif confidence >= 0.5:
                    conf_label = "MEDIUM"
                else:
                    conf_label = "LOW"

                print(f"  Result: {species} ({conf_label} confidence: {confidence:.0%})")
                print(f"  Reasoning: {reasoning}")
                print(f"  Latency: {data.get('total_latency_seconds', elapsed):.2f}s")
                print(f"  Chunks used: {data.get('chunks_used', 0)}")

                results.append({
                    "name": name,
                    "species": species,
                    "confidence": confidence,
                    "latency": data.get("total_latency_seconds", elapsed),
                })
            elif resp.status_code == 429:
                print(f"  Rate limited. Waiting...")
                time.sleep(10)
            else:
                print(f"  Error: HTTP {resp.status_code}")
                error = resp.json()
                print(f"  Detail: {error.get('detail', 'Unknown')}")

        except requests.Timeout:
            print(f"  Timed out (>35s)")
        except Exception as e:
            print(f"  Error: {e}")

        demo["name"] = name  # Restore
        print()

    # Summary
    print("=" * 70)
    print("  Demo Summary")
    print("=" * 70)

    if results:
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        identified = sum(1 for r in results if r["species"] != "DECLINED")

        print(f"  Queries run:        {len(results)}/{len(DEMO_QUERIES)}")
        print(f"  Species identified: {identified}")
        print(f"  Declined:           {len(results) - identified}")
        print(f"  Avg confidence:     {avg_confidence:.0%}")
        print(f"  Avg latency:        {avg_latency:.2f}s")

    print(f"\n  API Docs:     {api_url}/docs")
    print(f"  Frontend:     http://localhost:8501")
    print(f"  Dashboard:    http://localhost:8502")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
