import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from detection.cargo.merge_and_train import (
    ImageRecord,
    _copy_records,
    _split_v4_groups,
    _unique_destination_name,
    _validate_existing_v4_dataset,
    _validate_merged_dataset,
    _validate_label_file,
    _training_progress,
    _write_data_yaml,
    _write_split_manifest,
    audit_dataset,
)


class CargoMergeAndTrainTests(unittest.TestCase):
    def test_accepts_valid_class_zero_yolo_box(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            label = Path(temporary_dir) / "sample.txt"
            label.write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")

            lines, counts, errors = _validate_label_file(
                label, {0: "bulky_cargo"}
            )

        self.assertEqual(lines, ("0 0.5 0.5 0.4 0.6",))
        self.assertEqual(counts, {0: 1})
        self.assertEqual(errors, [])

    def test_rejects_unknown_class_and_out_of_bounds_boxes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            label = Path(temporary_dir) / "sample.txt"
            label.write_text(
                "1 0.5 0.5 0.4 0.4\n0 0.9 0.5 0.4 0.4\n",
                encoding="utf-8",
            )

            _, counts, errors = _validate_label_file(
                label, {0: "bulky_cargo"}
            )

        self.assertEqual(counts, {})
        self.assertEqual(len(errors), 2)

    def test_rejects_non_finite_box_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            label = Path(temporary_dir) / "sample.txt"
            label.write_text(
                "0 NaN 0.5 0.4 0.4\n", encoding="utf-8"
            )

            _, counts, errors = _validate_label_file(
                label, {0: "bulky_cargo"}
            )

        self.assertEqual(counts, {})
        self.assertEqual(errors[0]["error"], "box values must be finite numbers")

    def test_audit_reports_orphan_labels_without_editing_sources(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(
                    str(image_dir / "sample.jpg"),
                    np.zeros((16, 16, 3), dtype=np.uint8),
                )
            )
            (label_dir / "sample.txt").write_text(
                "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
            orphan = label_dir / "orphan.txt"
            orphan.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (root / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "train/images",
                        "nc": 1,
                        "names": {0: "bulky_cargo"},
                    }
                ),
                encoding="utf-8",
            )

            report = audit_dataset("sample", root)

            self.assertEqual(report["splits"]["train"]["image_count"], 1)
            self.assertEqual(report["splits"]["train"]["orphan_labels"], 1)
            self.assertEqual(
                report["splits"]["train"]["orphan_label_examples"],
                ["orphan.txt"],
            )
            self.assertTrue(orphan.is_file())

    def test_renames_colliding_image_stems_with_source_prefix(self):
        used_names = {"same.jpg"}

        renamed = _unique_destination_name("v4", "same.jpg", used_names)

        self.assertEqual(renamed, "v4_same.jpg")
        self.assertIn("v4_same.jpg", used_names)

    def test_splits_augmented_images_by_source_group_deterministically(self):
        records = []
        for group_index in range(5):
            for variant in range(3):
                image_name = f"source{group_index}.rf.variant{variant}.jpg"
                records.append(
                    ImageRecord(
                        source="v4",
                        split="train",
                        image=Path(image_name),
                        label=Path(image_name).with_suffix(".txt"),
                        sha256=f"{group_index:02d}{variant:02d}",
                        source_group=f"source{group_index}",
                        label_lines=("0 0.5 0.5 0.4 0.4",),
                    )
                )

        first, first_manifest, _ = _split_v4_groups(records, seed=42)
        second, second_manifest, _ = _split_v4_groups(records, seed=42)

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(
            {item.image.name: item.split for item in first},
            {item.image.name: item.split for item in second},
        )
        split_by_group = {}
        for item in first:
            split_by_group.setdefault(item.source_group, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))
        self.assertEqual(
            {split: sum(item.split == split for item in first)
             for split in ("train", "valid", "test")},
            {"train": 9, "valid": 3, "test": 3},
        )

    def test_merged_dataset_resolves_with_ultralytics(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for split, shade in (("train", 32), ("valid", 96)):
                image_dir = root / split / "images"
                label_dir = root / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                self.assertTrue(
                    cv2.imwrite(
                        str(image_dir / "sample.jpg"),
                        np.full((16, 16, 3), shade, dtype=np.uint8),
                    )
                )
                (label_dir / "sample.txt").write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )

            yaml_path = _write_data_yaml(root)
            report, resolved = _validate_merged_dataset(
                root, yaml_path, {"train": 1, "valid": 1, "test": 0}
            )

        self.assertEqual(report["class_names"], {"0": "bulky_cargo"})
        self.assertTrue(resolved["train"])
        self.assertTrue(resolved["val"])
        self.assertNotIn("test", resolved)

    def test_existing_v4_preparation_is_revalidated_without_recopying(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "source"
            source_records = []
            for group_index in range(3):
                image_name = f"group{group_index}.rf.original.jpg"
                image_path = source_root / "train" / "images" / image_name
                label_path = (
                    source_root
                    / "train"
                    / "labels"
                    / f"{Path(image_name).stem}.txt"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(
                    cv2.imwrite(
                        str(image_path),
                        np.full(
                            (16, 16, 3),
                            40 + group_index * 40,
                            dtype=np.uint8,
                        ),
                    )
                )
                label_path.write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
                source_records.append(
                    ImageRecord(
                        source="v4",
                        split="train",
                        image=image_path,
                        label=label_path,
                        sha256=hashlib.sha256(
                            image_path.read_bytes()
                        ).hexdigest(),
                        source_group=f"group{group_index}",
                        label_lines=("0 0.5 0.5 0.5 0.5",),
                    )
                )

            split_records, manifest_rows, _ = _split_v4_groups(
                source_records, seed=42
            )
            output_root = root / "prepared"
            _copy_records(split_records, output_root, {})
            _write_split_manifest(output_root, manifest_rows)
            yaml_path = _write_data_yaml(output_root)

            report, resolved, split_details = _validate_existing_v4_dataset(
                output_root, source_records
            )

        self.assertEqual(report["class_names"], {"0": "bulky_cargo"})
        self.assertTrue(resolved["train"])
        self.assertTrue(resolved["val"])
        self.assertEqual(split_details["group_count"], 3)

    def test_reads_actual_epochs_and_best_epoch_from_results_csv(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            results = Path(temporary_dir) / "results.csv"
            results.write_text(
                "epoch,metrics/mAP50-95(B)\n"
                "1,0.2\n"
                "2,0.5\n"
                "3,0.4\n",
                encoding="utf-8",
            )

            progress = _training_progress(temporary_dir, requested_epochs=5)

        self.assertEqual(progress["epochs_completed"], 3)
        self.assertEqual(progress["best_epoch"], 2)
        self.assertTrue(progress["stopped_early"])


if __name__ == "__main__":
    unittest.main()
