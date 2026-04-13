"""
Run this once before starting the app:
    python scripts/download_data.py

Downloads OFF taxonomy files into data/ directory.
These are bundled with the app — no API calls needed at runtime.
"""
import httpx
import json
import pathlib
import sys

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "additives.json": "https://static.openfoodfacts.org/data/taxonomies/additives.json",
    "ingredients.json": "https://static.openfoodfacts.org/data/taxonomies/ingredients.json",
}


def download(filename: str, url: str) -> None:
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  {filename} already exists — skipping. Delete to re-download.")
        return

    print(f"  Downloading {filename} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)

    # Verify it's valid JSON
    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  {filename} — {len(data):,} entries, {dest.stat().st_size / 1024:.0f} KB")


def main() -> None:
    print("NutriSnap — downloading OFF taxonomy data files")
    print(f"Destination: {DATA_DIR}\n")
    for filename, url in FILES.items():
        try:
            download(filename, url)
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)
    print("\nDone. Run the app now.")


if __name__ == "__main__":
    main()
