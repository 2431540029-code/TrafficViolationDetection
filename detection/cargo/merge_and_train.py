import argparse
import csv
import hashlib
import json
import logging
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_V3 = BASE_DIR / "dataset_cargo" / "roboflow-v3"
DEFAULT_V4 = BASE_DIR / "dataset_cargo" / "roboflow-v4"
DEFAULT_OUTPUT = BASE_DIR / "dataset_cargo" / "roboflow-combined"
DEFAULT_V4_ONLY_OUTPUT = BASE_DIR / "dataset_cargo" / "roboflow-v4-only-prepared"
DEFAULT_V4_MANIFEST = (
    BASE_DIR / "dataset_cargo" / "roboflow-v4-prepared" / "split_manifest.csv"
)
DEFAULT_WEIGHTS = BASE_DIR / "yolo11n.pt"
DEFAULT_COMBINED_RUN_NAME = "cargo_protrusion_combined"
DEFAULT_V4_ONLY_RUN_NAME = "cargo_protrusion_v4_only"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CANONICAL_CLASS = "bulky_cargo"
RECOGNIZED_CLASSES = {CANONICAL_CLASS, "protrusion", "xe_cong_kenh"}


@dataclass(frozen=True)
class ImageRecord:
    source: str
    split: str
    image: Path
    label: Path
    sha256: str
    source_group: str
    label_lines: tuple[str, ...]


def _class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        parsed = {int(class_id): str(name).strip().lower()
                  for class_id, name in names.items()}
    elif isinstance(names, list):
        parsed = {
            class_id: str(name).strip().lower()
            for class_id, name in enumerate(names)
        }
    else:
        return {}
    return parsed


def _configured_split_path(root, yaml_path, config, split):
    value = config.get(split)
    if not isinstance(value, str) or not value.strip():
        return None

    configured_root = config.get("path")
    if configured_root:
        configured_root = Path(configured_root)
        if not configured_root.is_absolute():
            configured_root = (yaml_path.parent / configured_root).resolve()
    else:
        configured_root = yaml_path.parent.resolve()
    path = Path(value)
    return (path if path.is_absolute() else configured_root / path).resolve()


def _validate_label_file(label_path, class_names):
    lines = tuple(
        line.strip()
        for line in label_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    errors = []
    class_counts = Counter()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("expected class_id and four box values")
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
            if class_id not in class_names:
                raise ValueError(f"class ID {class_id} is absent from data.yaml")
            if any(not math.isfinite(value) for value in box):
                raise ValueError("box values must be finite numbers")
            if any(value < 0 or value > 1 for value in box):
                raise ValueError("box values must be between 0 and 1")
            center_x, center_y, width, height = box
            if width <= 0 or height <= 0:
                raise ValueError("box width and height must be positive")
            if (
                center_x - width / 2 < 0
                or center_y - height / 2 < 0
                or center_x + width / 2 > 1
                or center_y + height / 2 > 1
            ):
                raise ValueError("box extends outside the image")
            class_counts[class_id] += 1
        except (ValueError, OverflowError) as error:
            errors.append({"line": line_number, "text": line, "error": str(error)})
    return lines, class_counts, errors


def audit_dataset(source, root):
    root = Path(root).resolve()
    result = {
        "root": str(root),
        "data_yaml": str(root / "data.yaml"),
        "class_names": {},
        "splits": {},
        "issues": [],
        "warnings": [],
        "images": [],
        "readme_image_count": None,
    }
    yaml_path = root / "data.yaml"
    if not root.is_dir():
        result["issues"].append(f"Dataset directory does not exist: {root}")
        return result
    if not yaml_path.is_file():
        result["issues"].append(f"Missing data.yaml: {yaml_path}")
        return result

    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        result["issues"].append(f"Cannot read data.yaml: {error}")
        return result
    if not isinstance(config, dict):
        result["issues"].append("data.yaml must contain a mapping.")
        return result

    class_names = _class_names(config)
    result["class_names"] = {
        str(class_id): name for class_id, name in class_names.items()
    }
    readme_path = root / "README.roboflow.txt"
    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"includes\s+(\d+)\s+images", readme_text, re.IGNORECASE)
        if match:
            result["readme_image_count"] = int(match.group(1))
    if config.get("nc") != len(class_names):
        result["issues"].append(
            f"{source}: nc={config.get('nc')} does not match names "
            f"({len(class_names)} class(es))."
        )
    if len(class_names) != 1 or class_names.get(0) not in RECOGNIZED_CLASSES:
        result["issues"].append(
            f"{source}: expected one recognized cargo class at ID 0; "
            f"found {class_names}."
        )
    if any(class_id != 0 for class_id in class_names):
        result["issues"].append(
            f"{source}: only class ID 0 can be mapped to {CANONICAL_CLASS}; "
            "other IDs require explicit label review."
        )

    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        split_report = {
            "image_count": 0,
            "label_count": 0,
            "valid_label_files": 0,
            "invalid_label_files": 0,
            "missing_label_images": 0,
            "orphan_labels": 0,
            "empty_label_files": 0,
            "object_count": 0,
            "class_counts": {},
            "missing_label_examples": [],
            "orphan_label_examples": [],
            "invalid_label_examples": [],
            "configured_image_dir": None,
            "configured_path_matches": None,
            "issues": [],
        }
        configured_path = _configured_split_path(root, yaml_path, config, split)
        if configured_path is not None:
            split_report["configured_image_dir"] = str(configured_path)
            split_report["configured_path_matches"] = (
                configured_path == image_dir.resolve()
            )
            if configured_path != image_dir.resolve():
                result["warnings"].append(
                    f"{source} {split}: data.yaml resolves to "
                    f"{configured_path}, but the physical split is {image_dir}."
                )

        if not image_dir.is_dir() and not label_dir.is_dir():
            split_report["status"] = "missing"
            result["warnings"].append(
                f"{source}: {split}/images and {split}/labels are missing."
            )
            result["splits"][split] = split_report
            continue
        if not image_dir.is_dir() or not label_dir.is_dir():
            message = (
                f"{source}: incomplete {split} split; both images and labels "
                "directories are required."
            )
            split_report["issues"].append(message)
            result["issues"].append(message)
            result["splits"][split] = split_report
            continue

        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_paths = sorted(label_dir.rglob("*.txt"))
        images_by_stem = defaultdict(list)
        labels_by_stem = defaultdict(list)
        for image_path in image_paths:
            images_by_stem[image_path.relative_to(image_dir).with_suffix("").as_posix()].append(image_path)
        for label_path in label_paths:
            labels_by_stem[label_path.relative_to(label_dir).with_suffix("").as_posix()].append(label_path)
        split_report["image_count"] = len(image_paths)
        split_report["label_count"] = len(label_paths)

        for stem, paths in images_by_stem.items():
            if len(paths) > 1:
                message = (
                    f"{source} {split}: multiple image files share label stem "
                    f"{stem}."
                )
                split_report["issues"].append(message)
                result["issues"].append(message)

        source_items = []
        for image_path in image_paths:
            relative_stem = image_path.relative_to(image_dir).with_suffix("").as_posix()
            matching_labels = labels_by_stem.get(relative_stem, [])
            if len(matching_labels) != 1:
                split_report["missing_label_images"] += 1
                if len(split_report["missing_label_examples"]) < 50:
                    split_report["missing_label_examples"].append(image_path.name)
                message = (
                    f"{source} {split}: image has "
                    f"{len(matching_labels)} matching labels: {image_path.name}"
                )
                split_report["issues"].append(message)
                result["issues"].append(message)
                continue

            label_path = matching_labels[0]
            try:
                lines, class_counts, errors = _validate_label_file(
                    label_path, class_names
                )
            except (OSError, UnicodeError) as error:
                errors = [{"line": 0, "text": "", "error": str(error)}]
                lines = ()
                class_counts = Counter()
            if errors:
                split_report["invalid_label_files"] += 1
                if len(split_report["invalid_label_examples"]) < 50:
                    split_report["invalid_label_examples"].append(
                        {"file": label_path.name, "errors": errors[:3]}
                    )
                message = (
                    f"{source} {split} invalid label {label_path.name}: "
                    f"{errors[:3]}"
                )
                split_report["issues"].append(message)
                result["issues"].append(message)
                continue

            split_report["valid_label_files"] += 1
            if not lines:
                split_report["empty_label_files"] += 1
            split_report["object_count"] += sum(class_counts.values())
            for class_id, count in class_counts.items():
                split_report["class_counts"][str(class_id)] = (
                    split_report["class_counts"].get(str(class_id), 0) + count
                )

            image_data = image_path.read_bytes()
            image = cv2.imdecode(
                np.frombuffer(image_data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if image is None:
                message = f"{source} {split}: unreadable image {image_path.name}"
                split_report["issues"].append(message)
                result["issues"].append(message)
                continue

            source_group = (
                image_path.stem.split(".rf.", 1)[0]
                if ".rf." in image_path.stem
                else f"{split}:{image_path.stem}"
            )
            source_items.append(
                ImageRecord(
                    source=source,
                    split=split,
                    image=image_path,
                    label=label_path,
                    sha256=hashlib.sha256(image_data).hexdigest(),
                    source_group=source_group,
                    label_lines=lines,
                )
            )

        image_stems = set(images_by_stem)
        split_report["orphan_labels"] = sum(
            len(paths)
            for stem, paths in labels_by_stem.items()
            if stem not in image_stems
        )
        if split_report["orphan_labels"]:
            split_report["orphan_label_examples"] = [
                path.name
                for stem, paths in labels_by_stem.items()
                if stem not in image_stems
                for path in paths
            ][:50]
            message = (
                f"{source} {split}: {split_report['orphan_labels']} label "
                "file(s) have no matching image."
            )
            split_report["issues"].append(message)
            result["issues"].append(message)
        split_report["status"] = "present" if image_paths else "empty"
        result["splits"][split] = split_report
        result["images"].extend(source_items)

    actual_image_count = sum(
        split.get("image_count", 0) for split in result["splits"].values()
    )
    if (
        result["readme_image_count"] is not None
        and actual_image_count != result["readme_image_count"]
    ):
        result["warnings"].append(
            f"{source}: Roboflow README reports "
            f"{result['readme_image_count']} images; found "
            f"{actual_image_count} image files in split folders."
        )
    if not result["splits"].get("train", {}).get("image_count"):
        result["issues"].append(f"{source}: no train images are available.")
    return result


def _duplicate_audit(datasets):
    hashes = defaultdict(list)
    groups = defaultdict(list)
    for dataset_name, report in datasets.items():
        for item in report["images"]:
            hashes[item.sha256].append(item)
            groups[(dataset_name, item.source_group)].append(item)

    exact_duplicates = []
    cross_split_duplicates = []
    for digest, items in hashes.items():
        if len(items) < 2:
            continue
        entry = {
            "sha256": digest,
            "files": [
                {
                    "dataset": item.source,
                    "split": item.split,
                    "name": item.image.name,
                }
                for item in items
            ],
        }
        exact_duplicates.append(entry)
        if len({item.split for item in items}) > 1:
            cross_split_duplicates.append(entry)

    augmented_groups = []
    cross_split_groups = []
    for (dataset_name, group), items in groups.items():
        splits = sorted({item.split for item in items})
        if len(items) > 1:
            entry = {
                "dataset": dataset_name,
                "source_group": group,
                "splits": splits,
                "files": [item.image.name for item in items],
            }
            augmented_groups.append(entry)
            if len(splits) > 1:
                cross_split_groups.append(entry)

    same_stem_across_sources = defaultdict(list)
    for dataset_name, report in datasets.items():
        for item in report["images"]:
            same_stem_across_sources[item.source_group].append(
                {
                    "dataset": dataset_name,
                    "split": item.split,
                    "name": item.image.name,
                }
            )
    cross_source_name_candidates = [
        {
            "source_group": group,
            "files": files,
            "cross_split": len({file["split"] for file in files}) > 1,
        }
        for group, files in same_stem_across_sources.items()
        if len({file["dataset"] for file in files}) > 1
    ]
    return {
        "exact_duplicate_groups": exact_duplicates,
        "exact_duplicates_across_splits": cross_split_duplicates,
        "augmentation_name_groups": augmented_groups,
        "augmentation_groups_across_splits": cross_split_groups,
        "same_source_stem_candidates_across_datasets": cross_source_name_candidates,
        "same_source_stem_candidates_across_splits": [
            item for item in cross_source_name_candidates if item["cross_split"]
        ],
    }


def _split_v4_groups(records, manifest_path=None, seed=42):
    groups = defaultdict(list)
    for item in records:
        groups[item.source_group].append(item)
    if len(groups) < 3:
        raise ValueError(
            "At least three independent v4 source groups are required to "
            "create train, validation and test splits."
        )

    assignments = {}
    if manifest_path is not None and Path(manifest_path).is_file():
        with Path(manifest_path).open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            manifest_rows = list(csv.DictReader(stream))
        expected_names = {item.image.name for item in records}
        manifest_names = [row.get("image", "") for row in manifest_rows]
        if (
            len(manifest_names) == len(expected_names)
            and set(manifest_names) == expected_names
        ):
            for row in manifest_rows:
                split = row.get("split")
                group = row.get("source_group")
                if split not in {"train", "valid", "test"} or not group:
                    assignments = {}
                    break
                assignments[row["image"]] = (group, split)

    if assignments:
        group_splits = defaultdict(set)
        for image_name, (group, split) in assignments.items():
            group_splits[group].add(split)
            source_item = next(item for item in records if item.image.name == image_name)
            if source_item.source_group != group:
                assignments = {}
                break
        if any(len(splits) != 1 for splits in group_splits.values()):
            assignments = {}

    if not assignments:
        group_names = sorted(groups)
        random.Random(seed).shuffle(group_names)
        test_count = max(1, round(len(group_names) * 0.17))
        valid_count = max(1, round(len(group_names) * 0.13))
        while test_count + valid_count >= len(group_names):
            if test_count >= valid_count and test_count > 1:
                test_count -= 1
            elif valid_count > 1:
                valid_count -= 1
            else:
                raise ValueError("Not enough v4 source groups for three splits.")
        group_assignments = {
            group: "test" for group in group_names[:test_count]
        }
        group_assignments.update({
            group: "valid"
            for group in group_names[test_count:test_count + valid_count]
        })
        group_assignments.update({
            group: "train"
            for group in group_names[test_count + valid_count:]
        })
        assignments = {
            item.image.name: (item.source_group, group_assignments[item.source_group])
            for item in records
        }

    split_records = []
    manifest_rows = []
    for item in records:
        group, split = assignments[item.image.name]
        split_records.append(replace(item, split=split))
        manifest_rows.append({
            "source_group": group,
            "split": split,
            "image": item.image.name,
            "label": item.label.name,
            "objects": len(item.label_lines),
        })

    split_audit = _duplicate_audit({"v4": {"images": split_records}})
    if split_audit["exact_duplicates_across_splits"]:
        raise ValueError("Identical v4 images would cross the generated splits.")
    if split_audit["augmentation_groups_across_splits"]:
        raise ValueError("A v4 source group would cross generated splits.")
    return split_records, manifest_rows, {
        "group_count": len(groups),
        "image_count_by_split": dict(Counter(item.split for item in split_records)),
        "seed": seed,
        "manifest_used": (
            str(manifest_path)
            if manifest_path is not None and Path(manifest_path).is_file()
            else None
        ),
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_report(v3_audit, v4_audit, duplicates):
    blockers = [
        issue
        for audit in (v3_audit, v4_audit)
        for issue in audit["issues"]
    ]
    if duplicates["exact_duplicates_across_splits"]:
        blockers.append("Identical image content occurs across dataset splits.")
    if duplicates["augmentation_groups_across_splits"]:
        blockers.append(
            "Filename-based augmentation groups occur across dataset splits."
        )
    if duplicates["same_source_stem_candidates_across_splits"]:
        blockers.append(
            "Potential same-source filename groups occur across dataset splits."
        )
    required_confirmations = []
    if v3_audit["class_names"].get("0") == "protrusion":
        required_confirmations.append(
            "Confirm that v3 'protrusion' labels and their bounding-box scope "
            "mean the same target as v4 motorcycle bulky-cargo labels."
        )
    return {
        "target_class": CANONICAL_CLASS,
        "datasets": {
            "v3": {key: value for key, value in v3_audit.items() if key != "images"},
            "v4": {key: value for key, value in v4_audit.items() if key != "images"},
        },
        "duplicates": duplicates,
        "blocking_issues": blockers,
        "required_human_confirmations": required_confirmations,
        "near_duplicate_limit": (
            "SHA-256 detects byte-identical images only. Filename stems are a "
            "heuristic for augmentation groups; perceptual near-duplicates "
            "were not exhaustively checked."
        ),
    }


def _print_audit(report):
    for dataset_name, dataset in report["datasets"].items():
        print(f"\n[{dataset_name}] {dataset['root']}")
        print(f"  classes: {dataset['class_names']}")
        for split, info in dataset["splits"].items():
            print(
                f"  {split}: images={info['image_count']}, "
                f"labels={info['label_count']}, "
                f"valid_labels={info['valid_label_files']}, "
                f"invalid_labels={info['invalid_label_files']}, "
                f"missing_labels={info['missing_label_images']}, "
                f"orphan_labels={info['orphan_labels']}, "
                f"objects={info['object_count']}, status={info['status']}"
            )
        for warning in dataset["warnings"]:
            print(f"  WARNING: {warning}")
        for issue in dataset["issues"]:
            print(f"  ERROR: {issue}")

    duplicates = report["duplicates"]
    print("\n[duplicate audit]")
    print(
        f"  exact duplicate groups={len(duplicates['exact_duplicate_groups'])}, "
        "exact duplicates across splits="
        f"{len(duplicates['exact_duplicates_across_splits'])}"
    )
    print(
        "  filename-based augmentation groups="
        f"{len(duplicates['augmentation_name_groups'])}, "
        "groups across splits="
        f"{len(duplicates['augmentation_groups_across_splits'])}"
    )
    print(
        "  possible same-source stems across datasets="
        f"{len(duplicates['same_source_stem_candidates_across_datasets'])}"
    )
    for blocker in report["blocking_issues"]:
        print(f"BLOCKER: {blocker}")
    for confirmation in report["required_human_confirmations"]:
        print(f"CONFIRMATION REQUIRED: {confirmation}")
    if not report["blocking_issues"] and not report["required_human_confirmations"]:
        print("Dataset audit passed.")


def _unique_destination_name(source, filename, used_names):
    candidate = filename
    if candidate.casefold() in used_names:
        candidate = f"{source}_{filename}"
    index = 2
    while candidate.casefold() in used_names:
        candidate = f"{source}_{Path(filename).stem}_{index}{Path(filename).suffix}"
        index += 1
    used_names.add(candidate.casefold())
    return candidate


def _copy_records(records, output_root, source_roots):
    copied_counts = Counter()
    renamed = []
    for split in ("train", "valid", "test"):
        used_names = set()
        for item in records:
            if item.split != split:
                continue
            image_name = _unique_destination_name(
                item.source, item.image.name, used_names
            )
            if image_name != item.image.name:
                renamed.append(
                    {
                        "source": item.source,
                        "split": split,
                        "original": item.image.name,
                        "renamed": image_name,
                    }
                )
            image_target = output_root / split / "images" / image_name
            label_target = output_root / split / "labels" / (
                Path(image_name).stem + ".txt"
            )
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.image, image_target)
            normalized_lines = [
                "0 " + " ".join(line.split()[1:])
                for line in item.label_lines
            ]
            label_target.write_text(
                "\n".join(normalized_lines) + ("\n" if normalized_lines else ""),
                encoding="utf-8",
            )
            copied_counts[split] += 1
    for source, source_root in source_roots.items():
        for metadata in ("README.dataset.txt", "README.roboflow.txt"):
            metadata_source = source_root / metadata
            if metadata_source.is_file():
                metadata_target = (
                    output_root / "source_metadata" / source / metadata
                )
                metadata_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata_source, metadata_target)
    return dict(copied_counts), renamed


def _write_data_yaml(output_root):
    config = {
        "path": output_root.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": {0: CANONICAL_CLASS},
    }
    test_images = output_root / "test" / "images"
    test_labels = output_root / "test" / "labels"
    if test_images.is_dir() and test_labels.is_dir():
        if any(path.is_file() for path in test_images.rglob("*")):
            config["test"] = "test/images"
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def _write_split_manifest(output_root, manifest_rows):
    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source_group", "split", "image", "label", "objects"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def _validate_merged_dataset(output_root, yaml_path, expected_counts):
    report = audit_dataset("combined", output_root)
    if report["issues"]:
        raise ValueError(
            "Merged dataset validation failed: "
            + "; ".join(report["issues"][:10])
        )
    if report["class_names"] != {"0": CANONICAL_CLASS}:
        raise ValueError(
            f"Unexpected merged class definition: {report['class_names']}"
        )
    for split in ("train", "valid", "test"):
        expected = expected_counts.get(split, 0)
        actual = report["splits"][split]["image_count"]
        if actual != expected:
            raise ValueError(
                f"Merged {split} image count mismatch: {actual} != {expected}."
            )
    split_duplicates = _duplicate_audit({"combined": report})
    if split_duplicates["exact_duplicates_across_splits"]:
        raise ValueError("Identical images were found across merged splits.")
    if split_duplicates["augmentation_groups_across_splits"]:
        raise ValueError(
            "A filename-based augmentation group crosses merged splits."
        )
    resolved = check_det_dataset(str(yaml_path), autodownload=False)
    if not resolved.get("train") or not resolved.get("val"):
        raise ValueError("Ultralytics did not resolve train and validation paths.")
    return report, resolved


def _validate_existing_v4_dataset(output_root, source_records):
    manifest_path = output_root / "split_manifest.csv"
    yaml_path = output_root / "data.yaml"
    if not manifest_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(
            "Existing v4-only dataset must contain split_manifest.csv and "
            "data.yaml."
        )
    expected_records, _, split_details = _split_v4_groups(
        source_records, manifest_path=manifest_path
    )
    expected_counts = split_details["image_count_by_split"]
    merged_audit, resolved_data = _validate_merged_dataset(
        output_root, yaml_path, expected_counts
    )
    expected_names = defaultdict(set)
    for item in expected_records:
        expected_names[item.split].add(item.image.name)
        image_path = output_root / item.split / "images" / item.image.name
        label_path = output_root / item.split / "labels" / item.label.name
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"Prepared v4 pair is missing: {item.image.name}"
            )
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != item.sha256:
            raise ValueError(f"Prepared image differs from source: {item.image.name}")
        output_label_lines = tuple(
            line.strip()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if output_label_lines != item.label_lines:
            raise ValueError(
                f"Prepared label differs from source: {item.label.name}"
            )

    for split in ("train", "valid", "test"):
        actual_names = {
            path.name
            for path in (output_root / split / "images").glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if actual_names != expected_names[split]:
            raise ValueError(
                f"Prepared {split} images do not match split_manifest.csv."
            )
    return merged_audit, resolved_data, split_details


def _choose_device(args):
    if args.device != "auto":
        return args.device
    if torch.cuda.is_available():
        return "0"
    message = (
        "CUDA is unavailable. CPU training may be very slow. "
        "Rerun with --allow-cpu to explicitly approve CPU training."
    )
    if args.allow_cpu:
        print(message)
        return "cpu"
    if sys.stdin.isatty():
        answer = input(message + " Continue on CPU? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            return "cpu"
    raise RuntimeError(message)


def _metrics_dict(metrics):
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def _training_progress(run_dir, requested_epochs):
    results_path = Path(run_dir) / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics did not write training metrics: {results_path}"
        )
    with results_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    metric_key = "metrics/mAP50-95(B)"
    valid_rows = [
        row for row in rows
        if row.get("epoch") not in (None, "")
        and row.get(metric_key) not in (None, "")
    ]
    if not valid_rows:
        raise ValueError(f"No valid training metrics in {results_path}.")
    best = max(valid_rows, key=lambda row: float(row[metric_key]))
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "stopped_early": len(rows) < requested_epochs,
    }


def _train(args, yaml_path, output_root, report):
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {weights}")
    project = (BASE_DIR / "runs" / "detect").resolve()
    if any(project.glob(f"{args.name}*")):
        raise FileExistsError(
            f"A run named {args.name} already exists. "
            "Choose a new --name rather than overwriting old results."
        )
    device = _choose_device(args)
    log_path = output_root / "merge_and_train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("cargo_merge_train")
    configuration = {
        "weights": str(weights),
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_requested": args.batch,
        "device": device,
        "workers": args.workers,
        "project": str(project),
        "name": args.name,
    }
    _write_json(output_root / "training_config.json", configuration)
    logger.info("Training configuration: %s", configuration)

    batches = [args.batch]
    while batches[-1] > 1:
        batches.append(max(1, batches[-1] // 2))
    result = None
    for batch in batches:
        try:
            model = YOLO(str(weights))
            logger.info("Starting YOLO11n training with batch=%d", batch)
            result = model.train(
                data=str(yaml_path),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch,
                device=device,
                workers=args.workers,
                project=str(project),
                name=args.name,
                exist_ok=False,
                patience=15,
                save=True,
                plots=True,
            )
            configuration["batch_used"] = batch
            break
        except RuntimeError as error:
            is_cuda_oom = "out of memory" in str(error).lower() and device != "cpu"
            if not is_cuda_oom or batch == batches[-1]:
                raise
            logger.warning(
                "CUDA out of memory at batch=%d; retrying with a smaller batch.",
                batch,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if result is None:
        raise RuntimeError("Training ended without a result.")
    run_dir = Path(result.save_dir)
    progress = _training_progress(run_dir, args.epochs)
    configuration.update(progress)
    _write_json(output_root / "training_config.json", configuration)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError(
            f"Training completed without expected checkpoints in {run_dir}."
        )

    model = YOLO(str(best))
    validation = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        plots=True,
    )
    evaluation = {
        "validation": _metrics_dict(validation),
        "test": None,
        "test_status": "not evaluated",
    }
    if args.evaluate_test:
        if not args.confirm_test_independent:
            raise ValueError(
                "Test evaluation requires --confirm-test-independent."
            )
        if "test" not in yaml.safe_load(yaml_path.read_text(encoding="utf-8")):
            evaluation["test_status"] = "no valid test split is configured"
        else:
            test = model.val(
                data=str(yaml_path),
                split="test",
                imgsz=args.imgsz,
                device=device,
                workers=args.workers,
                plots=True,
            )
            evaluation["test"] = _metrics_dict(test)
            evaluation["test_status"] = "evaluated after model selection"

    final_report = {
        **report,
        "training": configuration,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "metrics": evaluation,
        "metric_note": (
            "Precision, Recall and mAP are object-detection metrics on the "
            "specified split, not overall system accuracy."
        ),
    }
    _write_json(output_root / "merge_report.json", final_report)
    logger.info("Best checkpoint: %s", best)
    logger.info("Last checkpoint: %s", last)
    logger.info(
        "Training epochs completed: %d; best epoch: %d",
        progress["epochs_completed"],
        progress["best_epoch"],
    )
    logger.info("Validation metrics: %s", evaluation["validation"])
    if evaluation["test"] is not None:
        logger.info("Test metrics: %s", evaluation["test"])
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit, merge Roboflow cargo datasets and train YOLO11n. "
            "Source datasets are never modified."
        )
    )
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--v4", type=Path, default=DEFAULT_V4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Explicitly allow CPU training when CUDA is unavailable.",
    )
    parser.add_argument(
        "--confirm-v3-compatible",
        action="store_true",
        help=(
            "Confirm v3 protrusion labels and their box scope mean the same "
            "target as v4 motorcycle bulky-cargo labels."
        ),
    )
    parser.add_argument(
        "--confirm-ignore-orphan-labels",
        action="store_true",
        help=(
            "Confirm unmatched source label files were reviewed and may be "
            "omitted from the copied dataset; source files remain untouched."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test only after model selection.",
    )
    parser.add_argument(
        "--confirm-test-independent",
        action="store_true",
        help="Confirm test has not been used for model selection.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Inspect sources and write an audit report without copying/training.",
    )
    parser.add_argument(
        "--v4-only",
        action="store_true",
        help=(
            "Use only v4 and split its augmentation/source groups; do not "
            "include truck-focused v3 data."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and validate the dataset, but do not start training.",
    )
    parser.add_argument(
        "--train-existing",
        action="store_true",
        help=(
            "Train from an already prepared, validated --v4-only dataset "
            "without copying or changing its image/label files."
        ),
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            DEFAULT_V4_ONLY_OUTPUT if args.v4_only else DEFAULT_OUTPUT
        )
    if args.name is None:
        args.name = (
            DEFAULT_V4_ONLY_RUN_NAME if args.v4_only
            else DEFAULT_COMBINED_RUN_NAME
        )
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        parser.error("--epochs, --imgsz, --batch must be positive and workers nonnegative.")
    if args.evaluate_test and not args.confirm_test_independent:
        parser.error(
            "--evaluate-test requires --confirm-test-independent."
        )
    if args.train_existing and not args.v4_only:
        parser.error("--train-existing is only supported with --v4-only.")
    if args.train_existing and args.prepare_only:
        parser.error("--train-existing cannot be combined with --prepare-only.")
    if not args.audit_only and args.output.exists() and not args.train_existing:
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {args.output}"
        )
    if args.train_existing and not args.output.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset does not exist: {args.output}"
        )

    v4_audit = audit_dataset("v4", args.v4)
    if args.v4_only:
        duplicates = _duplicate_audit({"v4": v4_audit})
        report = {
            "mode": "v4-only",
            "target_class": CANONICAL_CLASS,
            "datasets": {
                "v4": {
                    key: value
                    for key, value in v4_audit.items()
                    if key != "images"
                }
            },
            "duplicates": duplicates,
            "blocking_issues": list(v4_audit["issues"]),
            "required_human_confirmations": [],
            "near_duplicate_limit": (
                "SHA-256 detects byte-identical images only. Filename stems "
                "are a heuristic for augmentation groups; perceptual "
                "near-duplicates were not exhaustively checked."
            ),
        }
        v3_audit = None
    else:
        v3_audit = audit_dataset("v3", args.v3)
        duplicates = _duplicate_audit({"v3": v3_audit, "v4": v4_audit})
        report = _make_report(v3_audit, v4_audit, duplicates)
    if args.confirm_ignore_orphan_labels and not args.v4_only:
        report["orphan_label_files_confirmed_ignored"] = {
            source: {
                split: info["orphan_label_examples"]
                for split, info in audit["splits"].items()
                if info["orphan_labels"]
            }
            for source, audit in (("v3", v3_audit), ("v4", v4_audit))
        }
        report["blocking_issues"] = [
            issue
            for issue in report["blocking_issues"]
            if "label file(s) have no matching image" not in issue
        ]
    report_path = args.output.with_name(args.output.name + "_audit.json")
    _write_json(report_path, report)
    _print_audit(report)
    print(f"\nAudit report: {report_path}")

    if args.audit_only:
        return
    if report["blocking_issues"]:
        raise SystemExit(
            "Stopped before copying or training because the audit has blocking "
            "issues. Review the audit report and fix the source data first."
        )
    if (
        report["required_human_confirmations"]
        and not args.confirm_v3_compatible
    ):
        raise SystemExit(
            "Stopped before copying or training. Review v3/v4 class semantics "
            "and bounding boxes, then rerun with --confirm-v3-compatible only "
            "if you confirm they represent the same target."
        )
    if args.output.exists() and not args.train_existing:
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {args.output}"
        )

    manifest_rows = None
    if args.v4_only:
        records, manifest_rows, split_details = _split_v4_groups(
            v4_audit["images"],
            manifest_path=(
                args.output / "split_manifest.csv"
                if args.train_existing
                else DEFAULT_V4_MANIFEST
                if args.v4.resolve() == DEFAULT_V4.resolve()
                else None
            ),
        )
        report["v4_group_split"] = split_details
    else:
        records = v3_audit["images"] + v4_audit["images"]
    if args.train_existing:
        merged_audit, resolved_data, split_details = (
            _validate_existing_v4_dataset(args.output, v4_audit["images"])
        )
        report["v4_group_split"] = split_details
        copied_counts = split_details["image_count_by_split"]
        report["validated_existing_images_by_split"] = copied_counts
        yaml_path = args.output / "data.yaml"
    else:
        copied_counts, renamed = _copy_records(
            records,
            args.output,
            (
                {"v4": args.v4.resolve()}
                if args.v4_only
                else {"v3": args.v3.resolve(), "v4": args.v4.resolve()}
            ),
        )
        if manifest_rows is not None:
            _write_split_manifest(args.output, manifest_rows)
        yaml_path = _write_data_yaml(args.output)
        merged_audit, resolved_data = _validate_merged_dataset(
            args.output, yaml_path, copied_counts
        )
        report["copied_images_by_split"] = copied_counts
        report["renamed_collisions"] = renamed
    report["merged_validation"] = {
        split: {
            "image_count": details["image_count"],
            "valid_label_files": details["valid_label_files"],
            "invalid_label_files": details["invalid_label_files"],
            "object_count": details["object_count"],
        }
        for split, details in merged_audit["splits"].items()
    }
    report["ultralytics_resolved_splits"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in resolved_data.items()
        if key in {"train", "val", "test", "names", "nc"}
    }
    _write_json(args.output / "merge_report.json", report)
    if args.prepare_only:
        print(
            f"Prepared dataset validated; training was not started: {args.output}"
        )
        return
    _train(args, yaml_path, args.output, report)


if __name__ == "__main__":
    main()
