"""Task 3 image-only data audit and duplicate-aware experiment split."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

TARGETS = ("gender", "usage")
USAGE_MERGE = {name: name for name in ("Casual", "Ethnic", "Formal", "Sports")}


def find_root(start=None):
    candidates = [Path(start).resolve()] if start else [Path.cwd(), Path(__file__).resolve().parent]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "A2_FashionDataset/FashionDataset/train/styles_train.csv").is_file():
                return parent
    raise FileNotFoundError("Dataset not found. Set repo_root to the extracted repository root.")


def prepare_image(source, size=(32, 24)):
    with Image.open(source) as original:
        picture = ImageOps.exif_transpose(original).convert("RGB")
        picture = picture.resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return np.asarray(picture, dtype=np.uint8)


def scan_images(directory):
    records = []
    for path in sorted(directory.glob("*.jpg")):
        if not path.stem.isdigit():
            continue
        record = {"id": int(path.stem), "file": path.name, "error": ""}
        try:
            with Image.open(path) as original:
                record.update(mode=original.mode, width=original.width, height=original.height)
                decoded = ImageOps.exif_transpose(original).convert("RGB")
                digest = hashlib.sha256(str(decoded.size).encode("ascii") + decoded.tobytes()).hexdigest()
                record["duplicate_group"] = digest
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            record["error"] = str(error)
        records.append(record)
    if not records:
        raise ValueError(f"No numeric-ID JPG images in {directory}")
    return pd.DataFrame(records)


def audit_data(repo_root, output_dir):
    root = find_root(repo_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = root / "A2_FashionDataset/FashionDataset"
    raw = pd.read_csv(dataset / "train/styles_train.csv")
    template = pd.read_csv(dataset / "test/styles_prediction.csv")
    required = {"id", *TARGETS}
    if not required.issubset(raw) or "id" not in template:
        raise ValueError("CSV schema does not contain the required Task 3 columns.")
    if raw.id.isna().any() or raw.id.duplicated().any() or template.id.duplicated().any():
        raise ValueError("CSV IDs must be present and unique; inspect the source data.")
    junk = [column for column in raw if column.startswith("Unnamed")]
    clean = raw.drop(columns=junk).copy()
    for target in TARGETS:
        clean[target] = clean[target].astype("string").str.strip().replace("", pd.NA)
    train_scan = scan_images(dataset / "train/images_train")
    test_scan = scan_images(dataset / "test/images_test")
    valid_train = train_scan.loc[train_scan.error.eq("")]
    valid_test = test_scan.loc[test_scan.error.eq("")]
    clean = clean.merge(valid_train[["id", "duplicate_group"]], on="id", how="inner", validate="one_to_one")
    clean = clean.sort_values("id").reset_index(drop=True)
    clean["image_path"] = clean.id.map(lambda identity: str(dataset / "train/images_train" / f"{identity}.jpg"))
    group_sizes = clean.groupby("duplicate_group").size()
    conflicting = clean.groupby("duplicate_group")[list(TARGETS)].nunique().gt(1).any(axis=1)
    stats = {
        "train_csv_rows": len(raw), "train_images_on_disk": len(train_scan),
        "usable_train_rows": len(clean), "train_missing_images": int((~raw.id.isin(train_scan.id)).sum()),
        "train_unreferenced_images": int((~train_scan.id.isin(raw.id)).sum()),
        "test_csv_rows": len(template), "test_images_on_disk": len(test_scan),
        "test_missing_or_corrupt_images": int((~template.id.isin(valid_test.id)).sum()),
        "missing_values_raw": {key: int(value) for key, value in raw.isna().sum().items()},
        "junk_columns": junk, "junk_rows": int(raw[junk].notna().any(axis=1).sum()) if junk else 0,
        "duplicate_groups_train": int(group_sizes.gt(1).sum()),
        "extra_duplicate_rows_train": int((group_sizes - 1).sum()),
        "duplicate_groups_with_target_conflicts": int(conflicting.sum()),
        "exact_train_test_duplicate_groups": len(set(clean.duplicate_group) & set(valid_test.duplicate_group)),
    }
    for name, scan in (("train", train_scan), ("test", test_scan)):
        stats[f"{name}_corrupt_images"] = int(scan.error.ne("").sum())
        stats[f"{name}_grayscale_images"] = int(scan["mode"].eq("L").sum())
        stats[f"{name}_irregular_sizes"] = int((scan.width.ne(60) | scan.height.ne(80)).sum())
        scan.to_csv(output_dir / f"audit_images_{name}.csv", index=False)
    (output_dir / "audit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    raw.loc[~raw.id.isin(clean.id)].to_csv(output_dir / "excluded_train_rows.csv", index=False)
    distributions = []
    for target in TARGETS:
        for label, count in clean[target].value_counts(dropna=False).items():
            distributions.append({"target": target, "label": str(label), "count": int(count), "share": count / len(clean)})
    pd.DataFrame(distributions).to_csv(output_dir / "class_distribution.csv", index=False)
    return clean, stats


def make_split(metadata, seed=42):
    data = metadata.copy()
    grouped = data.groupby("duplicate_group", sort=True)
    representatives = grouped[list(TARGETS)].first().fillna("<missing>")
    representatives["conflicting"] = grouped[list(TARGETS)].nunique().gt(1).any(axis=1)
    representatives["stratum"] = representatives.gender + "|" + representatives.usage
    assignments = {}
    generator = np.random.default_rng(seed)
    for _, section in representatives.groupby("stratum", sort=True):
        eligible = section.loc[~section.conflicting]
        identities = generator.permutation(eligible.index.to_numpy())
        evaluation_count = max(1, int(round(len(identities) * 0.15))) if len(identities) >= 3 else 0
        for index, identity in enumerate(identities):
            assignments[identity] = "holdout" if index < evaluation_count else "validation" if index < 2 * evaluation_count else "train"
    data["split"] = data.duplicate_group.map(assignments).fillna("train")
    if data.groupby("duplicate_group").split.nunique().max() != 1:
        raise AssertionError("Duplicate leakage between splits.")
    for target in TARGETS:
        train_labels = set(data.loc[data.split.eq("train"), target].dropna())
        all_labels = set(data[target].dropna())
        if train_labels != all_labels:
            raise AssertionError(f"Training does not cover all {target} labels.")
    return data


def load_pixels(metadata, size=(32, 24)):
    pixels = np.empty((len(metadata), *size, 3), dtype=np.uint8)
    for position, path in enumerate(metadata.image_path):
        pixels[position] = prepare_image(path, size)
    return pixels


def target_arrays(metadata, target, classes):
    labels = metadata.usage.map(lambda value: USAGE_MERGE.get(value, "Other") if pd.notna(value) else pd.NA) if target == "usage_5class" else metadata[target]
    indices = {name: index for index, name in enumerate(classes)}
    encoded = labels.map(indices)
    result = {}
    for split in ("train", "validation", "holdout"):
        positions = np.flatnonzero((metadata.split.eq(split) & labels.notna()).to_numpy())
        if len(positions) == 0:
            raise ValueError(f"Empty {target} {split} partition.")
        result[split] = (positions, encoded.iloc[positions].to_numpy(dtype=np.int64))
    return result
