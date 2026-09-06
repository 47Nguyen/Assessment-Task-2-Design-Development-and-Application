"""
Task 3: Check fashion images and prepare a duplicate-aware data split.

HOW TO RUN (from the project root):
    python -m tasks.task3_gender_usage.data --output-dir outputs/task3/data_check

This command audits the dataset and saves a split manifest; it does NOT train
a model. train.py imports the same functions and repeats these checks itself.
Choose a new output directory for each check so earlier results stay intact.
"""

# ---------------------------------------------------------------------------
# 1. Setup and Task 3 targets
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

TARGETS = ("gender", "usage")

# Used only in the optional five-class usage experiment. Other non-missing
# usage labels become "Other"; the official prediction still uses all labels.
USAGE_MERGE = {
    name: name for name in ("Casual", "Ethnic", "Formal", "Sports")
}


def find_root(start=None):
    """Find a parent folder containing the extracted assignment dataset."""
    if start is not None:
        candidates = [Path(start).resolve()]
    else:
        candidates = [Path.cwd(), Path(__file__).resolve().parent]

    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            training_csv = (
                parent / "A2_FashionDataset/FashionDataset/train/styles_train.csv"
            )
            if training_csv.is_file():
                return parent

    raise FileNotFoundError(
        "Dataset not found. Set repo_root to the extracted repository root."
    )


# ---------------------------------------------------------------------------
# 2. Prepare images the SAME way for training and prediction
# ---------------------------------------------------------------------------
def prepare_image(source, size=(32, 24)):
    """Return an RGB uint8 image; size is (height, width), not PIL's order.

    Pixel values stay in 0..255 here. The MLP's Rescaling layer performs the
    division by 255, so prediction must not divide them a second time.
    source may be a file path or an opened binary stream.
    """
    with Image.open(source) as original:
        picture = ImageOps.exif_transpose(original).convert("RGB")
        picture = picture.resize(
            (size[1], size[0]), Image.Resampling.BILINEAR
        )
        return np.asarray(picture, dtype=np.uint8)


def scan_images(directory):
    """Record image properties, read errors and exact decoded-image hashes.

    Equal hashes identify identical decoded RGB pixels at the same dimensions,
    not merely similar-looking products. Hash BEFORE resizing so preprocessing
    does not cause different images to be treated as exact duplicates.
    """
    records = []
    for path in sorted(Path(directory).glob("*.jpg")):
        if not path.stem.isdigit():
            continue

        record = {
            "id": int(path.stem),
            "file": path.name,
            "error": "",
            "mode": None,
            "width": None,
            "height": None,
            "duplicate_group": None,
        }
        try:
            with Image.open(path) as original:
                record["mode"] = original.mode
                record["width"] = original.width
                record["height"] = original.height
                decoded = ImageOps.exif_transpose(original).convert("RGB")
                image_bytes = str(decoded.size).encode("ascii") + decoded.tobytes()
                record["duplicate_group"] = hashlib.sha256(image_bytes).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            record["error"] = str(error)
        records.append(record)

    if not records:
        raise ValueError(f"No numeric-ID JPG images in {directory}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Audit the CSV files and images; keep usable training rows
# ---------------------------------------------------------------------------
def audit_data(repo_root, output_dir):
    """Save audit reports and return (clean_metadata, audit_statistics).

    Missing labels stay missing. A row without usage can still train gender.
    Test images are inspected only for data quality; their unknown labels are
    never used to train a model or select the best configuration.
    """
    root = find_root(repo_root)
    dataset = root / "A2_FashionDataset/FashionDataset"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(dataset / "train/styles_train.csv")
    template = pd.read_csv(dataset / "test/styles_prediction.csv")
    required_columns = {"id", *TARGETS}
    if not required_columns.issubset(raw.columns) or "id" not in template:
        raise ValueError("CSV schema does not contain the required Task 3 columns.")

    for frame in (raw, template):
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(
                "CSV IDs must be present and unique; inspect the source data."
            )

    junk_columns = [column for column in raw if column.startswith("Unnamed")]
    clean = raw.drop(columns=junk_columns).copy()
    for target in TARGETS:
        clean[target] = clean[target].astype("string").str.strip().replace("", pd.NA)

    train_scan = scan_images(dataset / "train/images_train")
    test_scan = scan_images(dataset / "test/images_test")
    valid_train = train_scan.loc[train_scan["error"].eq("")]
    valid_test = test_scan.loc[test_scan["error"].eq("")]

    # Join by ID, not by directory order. Missing or unreadable images are
    # excluded from training and recorded in excluded_train_rows.csv.
    clean = clean.merge(
        valid_train[["id", "duplicate_group"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    clean = clean.sort_values("id").reset_index(drop=True)
    clean["image_path"] = clean["id"].map(
        lambda identity: str(dataset / "train/images_train" / f"{identity}.jpg")
    )

    group_sizes = clean.groupby("duplicate_group").size()
    conflicting = (
        clean.groupby("duplicate_group")[list(TARGETS)]
        .nunique().gt(1).any(axis=1)
    )
    train_hashes = set(clean["duplicate_group"])
    test_hashes = set(valid_test["duplicate_group"])
    stats = {
        "train_csv_rows": len(raw),
        "train_images_on_disk": len(train_scan),
        "usable_train_rows": len(clean),
        "train_missing_images": int((~raw["id"].isin(train_scan["id"])).sum()),
        "train_unreferenced_images": int((~train_scan["id"].isin(raw["id"])).sum()),
        "test_csv_rows": len(template),
        "test_images_on_disk": len(test_scan),
        "test_missing_or_corrupt_images": int(
            (~template["id"].isin(valid_test["id"])).sum()
        ),
        "missing_values_raw": {
            key: int(value) for key, value in raw.isna().sum().items()
        },
        "junk_columns": junk_columns,
        "junk_rows": (
            int(raw[junk_columns].notna().any(axis=1).sum()) if junk_columns else 0
        ),
        "duplicate_groups_train": int(group_sizes.gt(1).sum()),
        "extra_duplicate_rows_train": int((group_sizes - 1).sum()),
        "duplicate_groups_with_target_conflicts": int(conflicting.sum()),
        "exact_train_test_duplicate_groups": len(train_hashes & test_hashes),
    }

    for name, scan in (("train", train_scan), ("test", test_scan)):
        stats[f"{name}_corrupt_images"] = int(scan["error"].ne("").sum())
        stats[f"{name}_grayscale_images"] = int(scan["mode"].eq("L").sum())
        stats[f"{name}_irregular_sizes"] = int(
            (scan["width"].ne(60) | scan["height"].ne(80)).sum()
        )
        scan.to_csv(output_dir / f"audit_images_{name}.csv", index=False)

    (output_dir / "audit.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    raw.loc[~raw["id"].isin(clean["id"])].to_csv(
        output_dir / "excluded_train_rows.csv", index=False
    )

    distributions = []
    for target in TARGETS:
        for label, count in clean[target].value_counts(dropna=False).items():
            distributions.append({
                "target": target,
                "label": str(label),
                "count": int(count),
                "share": count / len(clean),
            })
    pd.DataFrame(distributions).to_csv(
        output_dir / "class_distribution.csv", index=False
    )
    return clean, stats


# ---------------------------------------------------------------------------
# 4. Split duplicate groups, not individual images
# ---------------------------------------------------------------------------
def make_split(metadata, seed=42):
    """Create one reproducible train/validation/holdout split for both targets.

    Within each gender|usage group, roughly 15% of exact-duplicate groups go
    to holdout and 15% to validation. Tiny groups (fewer than 3) and groups
    with conflicting labels stay in training. Image percentages may therefore
    differ from 70/15/15; this is deliberately NOT a random split of rows.
    """
    if metadata.empty:
        raise ValueError("No usable training rows are available for splitting.")

    data = metadata.copy()
    grouped = data.groupby("duplicate_group", sort=True)
    representatives = grouped[list(TARGETS)].first().fillna("<missing>")
    representatives["conflicting"] = (
        grouped[list(TARGETS)].nunique().gt(1).any(axis=1)
    )
    representatives["stratum"] = (
        representatives["gender"] + "|" + representatives["usage"]
    )

    assignments = {}
    generator = np.random.default_rng(seed)
    for _, section in representatives.groupby("stratum", sort=True):
        eligible = section.loc[~section["conflicting"]]
        identities = generator.permutation(eligible.index.to_numpy())
        if len(identities) >= 3:
            evaluation_count = max(1, int(round(len(identities) * 0.15)))
        else:
            evaluation_count = 0

        for position, identity in enumerate(identities):
            if position < evaluation_count:
                assignments[identity] = "holdout"
            elif position < 2 * evaluation_count:
                assignments[identity] = "validation"
            else:
                assignments[identity] = "train"

    # Conflicting groups were not assigned above; retain them in training.
    data["split"] = data["duplicate_group"].map(assignments).fillna("train")
    if data.groupby("duplicate_group")["split"].nunique().max() != 1:
        raise AssertionError("Duplicate leakage between splits.")
    for target in TARGETS:
        train_labels = set(data.loc[data["split"].eq("train"), target].dropna())
        all_labels = set(data[target].dropna())
        if train_labels != all_labels:
            raise AssertionError(f"Training does not cover all {target} labels.")
    return data


# ---------------------------------------------------------------------------
# 5. Convert metadata into image arrays and target-specific labels
# ---------------------------------------------------------------------------
def load_pixels(metadata, size=(32, 24)):
    """Load uint8 pixels in exactly the same row order as metadata."""
    pixels = np.empty((len(metadata), *size, 3), dtype=np.uint8)
    for position, path in enumerate(metadata["image_path"]):
        pixels[position] = prepare_image(path, size)
    return pixels


def target_arrays(metadata, target, classes):
    """Return each split's (row positions, integer labels) for one target.

    Row positions select the matching images from load_pixels(). Drop a
    missing label only for this target, rather than discarding the entire row.
    Class order comes from training and must also be saved with the model.
    """
    if target == "usage_5class":
        labels = metadata["usage"].map(
            lambda value: USAGE_MERGE.get(value, "Other")
            if pd.notna(value) else pd.NA
        )
    else:
        labels = metadata[target]

    indices = {name: index for index, name in enumerate(classes)}
    encoded = labels.map(indices)
    if encoded.loc[labels.notna()].isna().any():
        raise ValueError(f"The saved class list does not cover every {target} label.")

    result = {}
    for split in ("train", "validation", "holdout"):
        usable = metadata["split"].eq(split) & labels.notna()
        positions = np.flatnonzero(usable.to_numpy())
        if len(positions) == 0:
            raise ValueError(f"Empty {target} {split} partition.")
        result[split] = (
            positions,
            encoded.iloc[positions].to_numpy(dtype=np.int64),
        )
    return result


# ---------------------------------------------------------------------------
# 6. Run a standalone data check (no TensorFlow or model training)
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="Extracted repository root.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/task3/data_check"),
        help="New folder for audit CSV files, audit.json and split_manifest.csv.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = find_root(args.repo_root)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"{output_dir} already contains data. Choose a new --output-dir."
        )

    print("TASK 3: DATA AUDIT AND DUPLICATE-AWARE SPLIT")
    print(f"Repository: {root}")
    metadata, stats = audit_data(root, output_dir)
    metadata = make_split(metadata, seed=args.seed)
    metadata.to_csv(output_dir / "split_manifest.csv", index=False)

    print(f"Usable training images: {stats['usable_train_rows']}")
    print(f"Exact-duplicate groups: {stats['duplicate_groups_train']}")
    print("\nImages per split:")
    print(metadata["split"].value_counts().to_string())
    print(f"\nSaved audit and split files to {output_dir}")
    print("No models were trained. Next: python -m tasks.task3_gender_usage.train")


if __name__ == "__main__":
    main()
