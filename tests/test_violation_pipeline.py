import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from core.violation_pipeline import (
    Detection,
    Vehicle,
    associate_with_vehicle,
    assign_cargo_to_vehicles,
    is_passenger_violation,
    merge_overlapping_detections,
    ViolationPipeline,
)
from detection.cargo.classes import normalize_cargo_classes
from detection.cargo.paths import latest_cargo_checkpoint
from detection.cargo.evaluate_cargo import calculate_iou, match_detections


class ViolationPipelineTests(unittest.TestCase):
    def test_normalizes_roboflow_protrusion_class_for_cargo_pipeline(self):
        class_ids, names = normalize_cargo_classes({0: "protrusion"})
        self.assertEqual(class_ids, [0])
        self.assertEqual(names, {0: "bulky_cargo"})

    def test_normalizes_v4_cargo_class_for_cargo_pipeline(self):
        class_ids, names = normalize_cargo_classes({0: "xe_cong_kenh"})
        self.assertEqual(class_ids, [0])
        self.assertEqual(names, {0: "bulky_cargo"})

    def test_rejects_unrelated_cargo_model_classes(self):
        with self.assertRaisesRegex(ValueError, "xe_cong_kenh"):
            normalize_cargo_classes({0: "backpack"})

    def test_default_checkpoint_prefers_v4_then_tracks_newest_v4_run(self):
        with TemporaryDirectory() as temporary_dir:
            run_root = Path(temporary_dir) / "runs" / "detect"
            legacy = run_root / "cargo_protrusion_v3_fixed" / "weights" / "best.pt"
            first = run_root / "cargo_protrusion_v4_only" / "weights" / "best.pt"
            second = run_root / "cargo_protrusion_v4_only2" / "weights" / "best.pt"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy")
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"old v4")
            second.write_bytes(b"new")
            self.assertEqual(latest_cargo_checkpoint(temporary_dir), second)

    def test_default_checkpoint_falls_back_to_legacy_when_v4_is_missing(self):
        with TemporaryDirectory() as temporary_dir:
            legacy = (
                Path(temporary_dir)
                / "runs"
                / "detect"
                / "cargo_protrusion_v3_fixed"
                / "weights"
                / "best.pt"
            )
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy")
            self.assertEqual(latest_cargo_checkpoint(temporary_dir), legacy)

    def test_default_checkpoint_prefers_relabelled_run_when_available(self):
        with TemporaryDirectory() as temporary_dir:
            run_root = Path(temporary_dir) / "runs" / "detect"
            previous = (
                run_root
                / "cargo_protrusion_v4_only"
                / "weights"
                / "best.pt"
            )
            relabelled = (
                run_root
                / "cargo_protrusion_relabel_v1"
                / "weights"
                / "best.pt"
            )
            previous.parent.mkdir(parents=True)
            relabelled.parent.mkdir(parents=True)
            previous.write_bytes(b"previous")
            relabelled.write_bytes(b"relabelled")

            self.assertEqual(
                latest_cargo_checkpoint(temporary_dir),
                relabelled,
            )

    def test_keeps_cargo_detection_when_no_motorcycle_is_associated(self):
        cargo = Detection((120, 10, 190, 60), 0.8, "bulky_cargo")
        unassigned = assign_cargo_to_vehicles([cargo], [])
        self.assertEqual(unassigned, [cargo])

    def test_unassigned_cargo_is_not_reported_or_drawn(self):
        pipeline = object.__new__(ViolationPipeline)
        pipeline.helmet_model = object()
        pipeline.passenger_model = object()
        pipeline.helmet_names = {0: "helmet", 1: "no_helmet"}
        pipeline.passenger_names = {0: "bike", 1: "rider", 2: "trippling"}
        pipeline.confidence = 0.35
        pipeline.no_helmet_confidence = 0.48
        pipeline.image_passenger_confidence = 0.2
        pipeline.cargo_model = type(
            "FakeCargoModel",
            (),
            {"predict": lambda self, *args, **kwargs: [object()]},
        )()
        pipeline.cargo_names = {0: "bulky_cargo"}
        pipeline.cargo_class_ids = [0]
        tree_false_positive = Detection(
            (120, 10, 190, 60),
            0.8,
            "bulky_cargo",
        )

        with (
            patch.object(
                ViolationPipeline,
                "_predict_image_detections",
                return_value=[],
            ),
            patch(
                "core.violation_pipeline.read_detections",
                return_value=[tree_false_positive],
            ),
        ):
            annotated, events, summary = pipeline.process_with_summary(
                np.zeros((120, 200, 3), dtype=np.uint8),
                track=False,
            )

        self.assertEqual(events, [])
        self.assertEqual(summary["cargo_detections"], 0)
        self.assertEqual(summary["violations"], [])
        self.assertTrue(np.array_equal(annotated[45, 155], [0, 0, 0]))

    def test_cargo_associated_with_motorcycle_remains_a_violation(self):
        pipeline = object.__new__(ViolationPipeline)
        pipeline.helmet_model = object()
        pipeline.passenger_model = object()
        pipeline.helmet_names = {0: "helmet", 1: "no_helmet"}
        pipeline.passenger_names = {0: "bike", 1: "rider", 2: "trippling"}
        pipeline.confidence = 0.35
        pipeline.no_helmet_confidence = 0.48
        pipeline.image_passenger_confidence = 0.2
        pipeline.cargo_model = type(
            "FakeCargoModel",
            (),
            {"predict": lambda self, *args, **kwargs: [object()]},
        )()
        pipeline.cargo_names = {0: "bulky_cargo"}
        pipeline.cargo_class_ids = [0]
        bike = Detection((10, 70, 90, 110), 0.9, "bike")
        cargo = Detection((35, 15, 65, 45), 0.8, "bulky_cargo")

        def fake_predict(_pipeline, model, _frame, _names, _targets, confidence=None):
            return [bike] if model is pipeline.passenger_model else []

        with (
            patch.object(
                ViolationPipeline,
                "_predict_image_detections",
                fake_predict,
            ),
            patch(
                "core.violation_pipeline.read_detections",
                return_value=[cargo],
            ),
        ):
            _, events, summary = pipeline.process_with_summary(
                np.zeros((120, 200, 3), dtype=np.uint8),
                track=False,
            )

        self.assertEqual(summary["cargo_detections"], 1)
        self.assertEqual(
            [event["violation"] for event in events],
            ["hang_cong_kenh"],
        )

    def test_assigns_cargo_to_nearby_motorcycle(self):
        cargo = Detection((35, 15, 65, 45), 0.8, "bulky_cargo")
        vehicle = Vehicle((10, 70, 90, 130), 1)
        unassigned = assign_cargo_to_vehicles([cargo], [vehicle])
        self.assertEqual(unassigned, [])
        self.assertEqual(vehicle.cargo_confidence, 0.8)

    def test_cargo_detection_box_is_drawn_red(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        cargo = Detection((5, 5, 30, 30), 0.8, "bulky_cargo")
        ViolationPipeline._draw_detection(frame, cargo)
        self.assertTrue(np.array_equal(frame[5, 10], [0, 0, 255]))

    def test_no_helmet_uses_a_higher_class_specific_confidence(self):
        pipeline = object.__new__(ViolationPipeline)
        pipeline.confidence = 0.35
        pipeline.no_helmet_confidence = 0.48
        helmet = Detection((1, 1, 10, 10), 0.36, "helmet")
        weak_no_helmet = Detection((11, 1, 20, 10), 0.47, "no_helmet")
        confident_no_helmet = Detection((21, 1, 30, 10), 0.48, "no_helmet")

        self.assertEqual(
            pipeline._filter_helmet_detections(
                [helmet, weak_no_helmet, confident_no_helmet]
            ),
            [helmet, confident_no_helmet],
        )

    def test_unassigned_no_helmet_is_not_reported_as_a_violation(self):
        pipeline = object.__new__(ViolationPipeline)
        pipeline.helmet_model = object()
        pipeline.passenger_model = object()
        pipeline.helmet_names = {0: "helmet", 1: "no_helmet"}
        pipeline.passenger_names = {0: "bike", 1: "rider", 2: "trippling"}
        pipeline.confidence = 0.35
        pipeline.no_helmet_confidence = 0.48
        pipeline.image_passenger_confidence = 0.2
        pipeline.cargo_model = None
        bike = Detection((10, 50, 90, 100), 0.9, "bike")
        far_no_helmet = Detection((150, 5, 170, 25), 0.8, "no_helmet")

        def fake_predict(_pipeline, model, _frame, _names, _targets, confidence=None):
            return [bike] if model is pipeline.passenger_model else [far_no_helmet]

        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        with patch.object(
            ViolationPipeline,
            "_predict_image_detections",
            fake_predict,
        ):
            annotated, events, summary = pipeline.process_with_summary(
                frame, track=False
            )

        self.assertEqual(events, [])
        self.assertEqual(summary["no_helmet"], 0)
        self.assertTrue(np.array_equal(annotated[5, 150], [0, 165, 255]))

    def test_assigns_rider_to_nearest_motorcycle(self):
        rider = Detection((40, 20, 60, 60), 0.9, "rider")
        vehicles = [
            Vehicle((10, 70, 50, 100), 1),
            Vehicle((55, 70, 95, 100), 2),
        ]
        self.assertEqual(associate_with_vehicle(rider, vehicles), 0)

    def test_rider_outside_expanded_vehicle_region_is_unassigned(self):
        rider = Detection((150, 10, 170, 50), 0.9, "rider")
        vehicles = [Vehicle((10, 70, 50, 100), 1)]
        self.assertIsNone(associate_with_vehicle(rider, vehicles))

    def test_overlapping_tile_detections_are_merged_per_class(self):
        detections = [
            Detection((10, 10, 50, 50), 0.8, "bike"),
            Detection((12, 12, 52, 52), 0.9, "bike"),
            Detection((12, 12, 52, 52), 0.7, "rider"),
        ]
        merged = merge_overlapping_detections(detections)
        self.assertEqual(
            [(d.class_name, d.confidence) for d in merged],
            [("bike", 0.9), ("rider", 0.7)],
        )

    def test_image_tiles_cover_edges_without_excessive_overlap(self):
        starts = ViolationPipeline._tile_starts(1000, 384, 0.2)
        self.assertEqual(starts, [0, 308, 616])

    def test_regular_sized_images_use_one_full_frame_pass(self):
        pipeline = object.__new__(ViolationPipeline)
        image = np.zeros((512, 768, 3), dtype=np.uint8)
        regions = pipeline._image_regions(image)
        self.assertEqual(len(regions), 1)
        self.assertIs(regions[0][2], image)

    def test_large_images_add_overlapping_tiles(self):
        pipeline = object.__new__(ViolationPipeline)
        image = np.zeros((768, 1537, 3), dtype=np.uint8)
        regions = pipeline._image_regions(image)
        self.assertEqual(len(regions), 4)
        self.assertEqual(
            ViolationPipeline._tile_starts(1537, 768, 0.2),
            [0, 384, 769],
        )

    def test_trippling_class_marks_vehicle_even_when_rider_count_is_low(self):
        vehicle = Vehicle((10, 10, 80, 80), 1, riders=2, trippling_confidence=0.8)
        self.assertTrue(is_passenger_violation(vehicle))

    def test_more_than_two_associated_riders_marks_violation(self):
        vehicle = Vehicle((10, 10, 80, 80), 1, riders=3)
        self.assertTrue(is_passenger_violation(vehicle))

    def test_two_riders_without_trippling_is_not_violation(self):
        vehicle = Vehicle((10, 10, 80, 80), 1, riders=2)
        self.assertFalse(is_passenger_violation(vehicle))

    def test_iou_and_unmatched_predictions_and_labels(self):
        first = (0.0, 0.0, 10.0, 10.0)
        self.assertEqual(calculate_iou(first, first), 1.0)
        predictions = [
            {"box": first, "confidence": 0.9},
            {"box": (20.0, 20.0, 30.0, 30.0), "confidence": 0.8},
        ]
        true_positives, false_positives, false_negatives = match_detections(
            predictions,
            [first, (40.0, 40.0, 50.0, 50.0)],
            0.5,
        )
        self.assertEqual(true_positives, 1)
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(false_negatives), 1)


if __name__ == "__main__":
    unittest.main()
