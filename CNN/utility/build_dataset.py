import os
import csv
import random
import requests
from pathlib import Path
from tqdm import tqdm

# === SETTINGS ===
PROJECT_ID = 200497
VALID_LICENSES = {'cc0', 'cc-by', 'cc-by-nc'}
BLOCKED_URL_PREFIX = 'https://static.inaturalist.org'
API_ENDPOINT = 'https://api.inaturalist.org/v1/observations'
OUTPUT_DIR = Path("../data/inat")
TRAIN_RATIO = 0.8
MAX_IMAGES = 1100

# === TARGET CLASSES===
MOULTING_CLASSES = {"pre-moult", "moulting", "post-moult", "exuviae"}


def get_taxon_group(ancestor_ids):
    if not ancestor_ids:
        return "Unknown"
    if 245097 in ancestor_ids:
        return "Chelicerata"
    elif 144128 in ancestor_ids:
        return "Myriapoda"
    elif 85493 in ancestor_ids:
        return "Crustacea"
    elif 372739 in ancestor_ids:
        return "Hexapoda"
    else:
        return "Unknown"


def fetch_observations(project_id, per_page=100, max_results=1000):
    results = []
    page = 1
    while len(results) < max_results:
        params = {
            'project_id': project_id,
            'per_page': per_page,
            'page': page,
            'license': ','.join(VALID_LICENSES),
            'photo_license': ','.join(VALID_LICENSES),
            'order': 'desc',
            'order_by': 'created_at'
        }
        r = requests.get(API_ENDPOINT, params=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            break
        results.extend(data["results"])
        if len(data["results"]) < per_page:
            break
        page += 1
    return results[:max_results]


def download_moulting_images(observations, base_dir, train_ratio):
    count_per_class = {cls: 0 for cls in MOULTING_CLASSES}
    skipped = 0
    metadata_records = []

    for obs in tqdm(observations, desc="Processing Moulting Stage"):
        photos = obs.get("photos", [])
        valid_photos = [p for p in photos if p.get("license_code") in VALID_LICENSES and not p.get("url", "").startswith(BLOCKED_URL_PREFIX)]
        if not valid_photos:
            continue

        moulting_stage = None
        for f in obs.get("ofvs", []):
            if f.get("name_ci", "").lower() == "moulting stage" and f.get("value_ci"):
                moulting_stage = f["value_ci"].strip().lower()
                break

        if moulting_stage not in MOULTING_CLASSES:
            skipped += 1
            continue

        taxon = obs.get("taxon", {})
        taxon_id = taxon.get("id", None)
        taxon_name = taxon.get("name", "Unknown")
        ancestor_ids = taxon.get("ancestor_ids", [])
        taxon_group = get_taxon_group(ancestor_ids)

        split = "train" if random.random() < train_ratio else "val"
        class_dir = base_dir / split / moulting_stage
        class_dir.mkdir(parents=True, exist_ok=True)

        img_url = valid_photos[0]['url'].replace('square', 'medium')
        try:
            img_data = requests.get(img_url, timeout=10).content
        except Exception:
            continue

        filename = f"{obs['id']}.jpg"
        img_path = class_dir / filename
        with open(img_path, "wb") as f:
            f.write(img_data)

        count_per_class[moulting_stage] += 1

        metadata_records.append({
            "filename": filename,
            "stage": moulting_stage,
            "split": split,
            "taxon_id": taxon_id,
            "taxon_name": taxon_name,
            "taxon_group": taxon_group,
            "observation_id": obs.get("id")
        })

    DATA_DIR = (Path(__file__).parent / "../../data").resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "annotated_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["filename", "stage", "split", "taxon_id", "taxon_name", "taxon_group", "observation_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    return count_per_class, skipped, csv_path


def build_moulting_stage_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = fetch_observations(PROJECT_ID, max_results=MAX_IMAGES)
    counts, skipped, csv_path = download_moulting_images(observations, OUTPUT_DIR, TRAIN_RATIO)

    print("\n Image count per class:")
    for stage, count in counts.items():
        print(f"- {stage}: {count} images")
    print(f"Skipped {skipped} observations with missing or invalid stage.")
    print(f"Metadata saved to {csv_path.resolve()}")


if __name__ == "__main__":
    build_moulting_stage_dataset()
