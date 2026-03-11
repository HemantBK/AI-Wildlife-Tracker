"""
First-Time Setup Script
Validates the environment, checks dependencies, and guides users
through the initial setup process.

Usage:
  python scripts/setup.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_check(name, ok, message=""):
    status = "PASS" if ok else "FAIL"
    icon = "+" if ok else "X"
    msg = f" — {message}" if message else ""
    print(f"  [{icon}] {name}: {status}{msg}")
    return ok


def check_python():
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    return print_check(
        "Python version",
        ok,
        f"{version.major}.{version.minor}.{version.micro}"
        + ("" if ok else " (need 3.10+)")
    )


def check_pip_packages():
    required = ["fastapi", "chromadb", "sentence_transformers", "yaml", "dotenv"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            # Try alternate names
            alt = {"yaml": "pyyaml", "dotenv": "python-dotenv"}
            try:
                __import__(alt.get(pkg, pkg))
            except ImportError:
                missing.append(pkg)

    if missing:
        return print_check(
            "Python packages",
            False,
            f"Missing: {', '.join(missing)}. Run: pip install -r requirements.txt"
        )
    return print_check("Python packages", True, "All core packages installed")


def check_env_file():
    env_path = Path(".env")
    example_path = Path(".env.example")

    if env_path.exists():
        return print_check(".env file", True, "Found")

    if example_path.exists():
        print_check(".env file", False, "Not found. Creating from .env.example...")
        shutil.copy(example_path, env_path)
        print("    Created .env — please edit it with your API keys.")
        return False
    else:
        return print_check(".env file", False, "Neither .env nor .env.example found")


def check_groq_key():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "")
    ok = bool(key) and not key.startswith("your_")
    return print_check(
        "Groq API key",
        ok,
        "Configured" if ok else "Not set. Get a free key at https://console.groq.com"
    )


def check_data():
    chunks_dir = Path("data/chunks")
    if chunks_dir.exists() and list(chunks_dir.glob("*.json")):
        count = len(list(chunks_dir.glob("*.json")))
        return print_check("Chunk data", True, f"{count} chunk file(s)")
    return print_check(
        "Chunk data",
        False,
        "No chunks found. Run: make data-pipeline"
    )


def check_indexes():
    chroma = Path("data/chroma_db")
    bm25 = Path("data/bm25_index.pkl")

    chroma_ok = chroma.exists() and any(chroma.iterdir()) if chroma.exists() else False
    bm25_ok = bm25.exists()

    if chroma_ok and bm25_ok:
        return print_check("Search indexes", True, "ChromaDB + BM25 ready")
    elif chroma_ok:
        return print_check("Search indexes", False, "ChromaDB OK, BM25 missing. Run: make build-indexes")
    elif bm25_ok:
        return print_check("Search indexes", False, "BM25 OK, ChromaDB missing. Run: make build-indexes")
    else:
        return print_check("Search indexes", False, "Not built. Run: make build-indexes")


def check_ollama():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return print_check("Ollama", True, f"Running. Models: {', '.join(models[:3])}")
    except Exception:
        pass
    return print_check(
        "Ollama",
        False,
        "Not running (optional — needed only for local inference mode)"
    )


def check_docker():
    docker = shutil.which("docker")
    compose = shutil.which("docker-compose") or shutil.which("docker")
    ok = docker is not None
    return print_check(
        "Docker",
        ok,
        "Installed" if ok else "Not found (optional — needed for docker-compose deployment)"
    )


def main():
    print_header("Wildlife Tracker - Environment Setup Check")

    results = []

    # Critical checks
    print("\n--- Critical ---")
    results.append(check_python())
    results.append(check_pip_packages())
    results.append(check_env_file())
    results.append(check_groq_key())

    # Data checks
    print("\n--- Data Pipeline ---")
    results.append(check_data())
    results.append(check_indexes())

    # Optional checks
    print("\n--- Optional ---")
    results.append(check_ollama())
    results.append(check_docker())

    # Summary
    passed = sum(results)
    total = len(results)
    print_header(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("\n  All checks passed! You're ready to go.")
        print("  Start the app with: make api  (then make frontend)")
        print("  Or with Docker:     docker-compose up --build")
    else:
        print("\n  Some checks failed. Follow the suggestions above.")
        print("  Critical items must be fixed before running the app.")

    print()
    return 0 if all(results[:4]) else 1  # Fail only on critical checks


if __name__ == "__main__":
    sys.exit(main())
