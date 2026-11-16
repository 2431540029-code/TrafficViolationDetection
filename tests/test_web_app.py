import io
import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from pillow_heif import from_pillow
from PIL import Image

from web.app import create_app


class FakePipeline:
    def __init__(self):
        self.cargo_detections = 0
        self.cargo_unassigned_detections = 0
        self.cargo_enabled = False

    def process_with_summary(self, image, track=True):
        self.track_enabled = track
        summary = {
            "helmet_on": 1,
            "no_helmet": 1,
            "no_helmet_detections": 1,
            "no_helmet_unassigned_detections": 0,
            "motorcycles": 1,
            "riders": 3,
            "riders_detected": 3,
            "riders_unassigned": 0,
            "passenger_violations": 1,
            "cargo_detections": self.cargo_detections,
            "cargo_unassigned_detections": self.cargo_unassigned_detections,
            "cargo_enabled": self.cargo_enabled,
            "total_detections": 6 + self.cargo_detections,
            "violation_count": 1,
            "violations": [
                {
                    "track_id": 7,
                    "violation": "cho_qua_so_nguoi",
                    "passenger_count": 3,
                    "confidence": "0.91",
                }
            ],
        }
        return image.copy(), summary["violations"], summary


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": root / "results",
                "HELMET_MODEL": root / "helmet.pt",
                "PASSENGER_MODEL": root / "passenger.pt",
                "CARGO_MODEL": root / "cargo.pt",
                "PIPELINE": FakePipeline(),
            }
        )
        self.client = app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dashboard_renders_model_status_and_upload_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Tải ảnh để nhận diện".encode(), response.data)
        self.assertIn("THIẾU CHECKPOINT".encode(), response.data)
        self.assertIn("multipart/form-data".encode(), response.data)
        self.assertIn(b'class="history-section"', response.data)
        self.assertIn(b'class="history-empty"', response.data)
        self.assertIn(b'id="violation-models-table"', response.data)
        self.assertIn(
            "Độ tin cậy nhận diện còn hạn chế; kết quả cần được kiểm chứng trong điều kiện thực tế.".encode(),
            response.data,
        )
        self.assertIn(
            "Trạng thái checkpoint chỉ xác nhận tệp trọng số tồn tại".encode(),
            response.data,
        )
        self.assertIn("Chưa có checkpoint Cargo".encode(), response.data)
        self.assertNotIn("Đã tích hợp · thử nghiệm".encode(), response.data)
        self.assertIn(b".heic,.heif", response.data)
        self.assertNotIn(b'id="model-performance-table"', response.data)
        self.assertNotIn(b'id="model-versions-table"', response.data)
        self.assertNotIn("Bảng 1".encode(), response.data)
        self.assertNotIn("Bảng 2".encode(), response.data)
        self.assertNotIn("Bảng 3".encode(), response.data)
        self.assertIn("Mô hình AI theo từng loại vi phạm".encode(), response.data)
        self.assertNotIn("biển số".encode(), response.data.lower())

    def test_dashboard_marks_selected_cargo_v6_as_integrated(self):
        root = Path(self.temp_dir.name)
        checkpoint = (
            root
            / "cargo_protrusion_v6_yolo11n"
            / "weights"
            / "best.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        helmet_checkpoint = root / "helmet.pt"
        passenger_checkpoint = root / "passenger.pt"
        helmet_checkpoint.touch()
        passenger_checkpoint.touch()
        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": root / "v6-results",
                "HELMET_MODEL": helmet_checkpoint,
                "PASSENGER_MODEL": passenger_checkpoint,
                "CARGO_MODEL": checkpoint,
            }
        )

        try:
            response = app.test_client().get("/")
        finally:
            app.extensions["comparison_executor"].shutdown(wait=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("YOLO11n fine-tune trên Roboflow v6".encode(), response.data)
        self.assertIn("Đã tích hợp".encode(), response.data)
        self.assertIn("CÓ CHECKPOINT".encode(), response.data)
        self.assertIn("Có checkpoint".encode(), response.data)
        self.assertIn(
            "Độ tin cậy nhận diện còn hạn chế; kết quả cần được kiểm chứng trong điều kiện thực tế.".encode(),
            response.data,
        )

    def test_ai_comparison_page_renders_registered_models(self):
        response = self.client.get("/ai-comparison")
        self.assertEqual(response.status_code, 200)
        self.assertIn("So sánh hiệu năng".encode(), response.data)
        self.assertIn("YOLO11n".encode(), response.data)
        self.assertIn("YOLOv8s".encode(), response.data)
        self.assertIn("YOLO26n".encode(), response.data)
        for model_id in (
            "helmet_yolo11n",
            "helmet_yolov8s",
            "helmet_yolo26n",
            "passenger_yolo11n",
            "passenger_yolov8s",
            "passenger_yolo26n",
            "cargo_yolo11n",
            "cargo_yolov8s",
            "cargo_yolo26n",
        ):
            self.assertIn(model_id.encode(), response.data)
        self.assertIn(b'id="chart-precision"', response.data)
        self.assertIn(b'id="chart-latency"', response.data)
        self.assertIn(b'id="model-catalog"', response.data)
        catalog_json = re.search(
            rb'<script id="model-catalog" type="application/json">(.*?)</script>',
            response.data,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(catalog_json)
        cargo_models = [
            model
            for model in json.loads(catalog_json.group(1))
            if model["task"] == "cargo"
        ]
        cargo_yolo11n = next(
            model for model in cargo_models if model["id"] == "cargo_yolo11n"
        )
        self.assertEqual(cargo_yolo11n["classes"], ["bulky_cargo"])
        self.assertEqual(
            cargo_yolo11n["dataset_label"],
            "Roboflow Cargo v2 · gắn nhãn lại, chia theo cảnh",
        )
        self.assertGreaterEqual(
            cargo_yolo11n["training_metrics"]["epoch"],
            1,
        )
        self.assertLessEqual(
            cargo_yolo11n["training_metrics"]["epoch"],
            50,
        )
        self.assertEqual(
            cargo_yolo11n["training_metrics"]["source"],
            "validation",
        )
        self.assertGreaterEqual(
            cargo_yolo11n["training_metrics"]["map50_95"],
            0,
        )
        self.assertLessEqual(
            cargo_yolo11n["training_metrics"]["map50_95"],
            1,
        )
        self.assertEqual(
            cargo_yolo11n["splits"]["val"]["source_group_count"],
            6,
        )
        self.assertFalse(cargo_yolo11n["splits"]["test"]["available"])
        self.assertIn(
            "đã được dùng",
            cargo_yolo11n["splits"]["test"]["error"],
        )

    def test_ai_comparison_rejects_missing_weights_and_invalid_model(self):
        missing_weights = self.client.post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["cargo_yolo11n"],
                "split": "val",
                "confidence": 0.25,
                "repeats": 1,
            },
        )
        self.assertEqual(missing_weights.status_code, 409)

        invalid_model = self.client.post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["C:\\secret\\model.pt"],
                "split": "val",
                "confidence": 0.25,
                "repeats": 1,
            },
        )
        self.assertEqual(invalid_model.status_code, 400)

    def test_ai_comparison_does_not_use_partial_training_checkpoint(self):
        root = Path(self.temp_dir.name)
        partial_weights = root / "cargo-yolov8s" / "weights" / "best.pt"
        partial_weights.parent.mkdir(parents=True)
        partial_weights.touch()
        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": root / "partial-results",
                "CARGO_YOLOV8S_MODEL": partial_weights,
            }
        )

        response = app.test_client().post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["cargo_yolov8s"],
                "split": "test",
                "confidence": 0.25,
                "repeats": 1,
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("huấn luyện", response.get_json()["error"].lower())

    def test_ai_comparison_blocks_previously_used_cargo_test_split(self):
        root = Path(self.temp_dir.name)
        weights = root / "cargo.pt"
        weights.touch()
        app = create_app({
            "TESTING": True,
            "RESULTS_DIR": root / "test-split-results",
            "CARGO_MODEL": weights,
        })

        response = app.test_client().post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["cargo_yolo11n"],
                "split": "test",
                "confidence": 0.25,
                "repeats": 1,
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("đã được dùng", response.get_json()["error"])

    def test_ai_comparison_runs_in_background_and_persists_results(self):
        root = Path(self.temp_dir.name)
        weights = root / "cargo.pt"
        weights.touch()

        extra_weights = root / "cargo-yolov8s" / "weights" / "best.pt"
        extra_weights.parent.mkdir(parents=True)
        extra_weights.touch()
        (extra_weights.parents[1] / "comparison-ready.txt").touch()

        def fake_runner(
            model_path, task_id, model_id, split, confidence, repeats, job_id
        ):
            return {
                "model_id": model_id,
                "task": task_id,
                "task_label": "Hàng hóa cồng kềnh",
                "model_name": model_id,
                "weight_file": model_path.name,
                "dataset": "Roboflow Cargo v4 · chia theo cảnh",
                "split": split,
                "evaluated_at": "2026-10-27T00:00:00+00:00",
                "precision": 0.6,
                "recall": 0.4,
                "map50": 0.5,
                "map50_95": 0.25,
                "inference_ms": 12.5,
                "inference_min_ms": 10.0,
                "inference_max_ms": 15.0,
                "model_size_mb": 5.2,
                "image_count": 32,
                "object_count": 59,
                "orphan_label_count": 1,
                "confidence": confidence,
                "repeats": repeats,
                "image_size": 640,
                "device": "CPU",
                "timing_scope": "Unit test",
                "status": "Đã đánh giá",
            }

        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": root / "comparison-results",
                "HELMET_MODEL": root / "helmet.pt",
                "PASSENGER_MODEL": root / "passenger.pt",
                "CARGO_MODEL": weights,
                "CARGO_YOLOV8S_MODEL": extra_weights,
                "COMPARISON_RUNNER": fake_runner,
            }
        )
        client = app.test_client()
        response = client.post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["cargo_yolo11n", "cargo_yolov8s"],
                "split": "val",
                "confidence": 0.25,
                "repeats": 2,
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["id"]

        for _ in range(100):
            status = client.get(f"/api/ai-comparison/jobs/{job_id}")
            self.assertEqual(status.status_code, 200)
            job = status.get_json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "completed")
        self.assertEqual(
            [result["model_id"] for result in job["results"]],
            ["cargo_yolo11n", "cargo_yolov8s"],
        )
        self.assertEqual(job["results"][0]["map50"], 0.5)
        history = client.get("/api/ai-comparison/history?task=cargo")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.get_json()["results"]), 2)
        self.assertEqual(history.get_json()["results"][0]["inference_ms"], 12.5)

    def test_upload_returns_result_and_downloadable_annotated_image(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(encoded.tobytes()), "road.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="results" id="results"', response.data)
        self.assertIn("TỔNG BOUNDING BOX".encode(), response.data)
        self.assertIn("SỰ KIỆN VI PHẠM".encode(), response.data)
        self.assertIn(b'class="notice"', response.data)
        self.assertIn(b'class="history-row"', response.data)
        self.assertIn("Ngưỡng nhận diện không đội mũ hiện là 48%".encode(), response.data)
        self.assertFalse(self.client.application.config["PIPELINE"].track_enabled)
        download_path = re.search(
            rb'href="(/download/[^"]+)"', response.data
        ).group(1).decode()
        detail_path = re.search(
            rb'href="(/history/[^"]+)"', response.data
        ).group(1).decode()
        download = self.client.get(download_path)
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.headers["Content-Disposition"].startswith("attachment"))
        download.close()
        detail = self.client.get(detail_path)
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"road.png", detail.data)

        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": self.client.application.config["RESULTS_DIR"],
                "HISTORY_DB": self.client.application.config["HISTORY_DB"],
                "HELMET_MODEL": Path("missing-helmet.pt"),
                "PASSENGER_MODEL": Path("missing-passenger.pt"),
                "CARGO_MODEL": Path("missing-cargo.pt"),
            }
        )
        with app.test_client() as second_client:
            persisted = second_client.get("/")
            self.assertIn(b'class="history-row"', persisted.data)

        delete_path = re.search(
            rb'action="(/history/[^"]+/delete)"', response.data
        ).group(1).decode()
        deleted = self.client.post(delete_path)
        self.assertEqual(deleted.status_code, 302)
        self.assertNotIn(b'class="history-row"', self.client.get("/").data)

    def test_result_uses_configured_no_helmet_threshold(self):
        self.client.application.config["NO_HELMET_CONFIDENCE"] = 0.61
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(encoded.tobytes()), "threshold.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Ngưỡng nhận diện không đội mũ hiện là 61%".encode(), response.data)

    def test_unassigned_cargo_is_explicitly_not_confirmed_as_a_violation(self):
        pipeline = self.client.application.config["PIPELINE"]
        pipeline.cargo_enabled = True
        pipeline.cargo_detections = 1
        pipeline.cargo_unassigned_detections = 1
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(encoded.tobytes()), "cargo.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cargo được giữ lại trong kết quả".encode(), response.data)
        self.assertIn("Model quét toàn ảnh, không lọc theo loại xe".encode(), response.data)
        self.assertNotIn(
            "Chưa đủ căn cứ xác định vi phạm cho các phát hiện này".encode(),
            response.data,
        )

    def test_empty_detection_result_shows_no_violation_message(self):
        class NoDetectionPipeline:
            def process_with_summary(self, image, track=True):
                summary = {
                    "total_detections": 0,
                    "helmet_on": 0,
                    "no_helmet": 0,
                    "no_helmet_detections": 0,
                    "no_helmet_unassigned_detections": 0,
                    "motorcycles": 0,
                    "riders": 0,
                    "riders_detected": 0,
                    "riders_unassigned": 0,
                    "passenger_violations": 0,
                    "cargo_detections": 0,
                    "cargo_unassigned_detections": 0,
                    "cargo_enabled": True,
                    "violation_count": 0,
                    "violations": [],
                }
                return image.copy(), [], summary

        self.client.application.config["PIPELINE"] = NoDetectionPipeline()
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(encoded.tobytes()), "empty.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="empty-result"', response.data)
        self.assertIn("Không phát hiện hành vi vi phạm".encode(), response.data)
        self.assertIn("<strong>0</strong>".encode(), response.data)

    def test_repeated_upload_filename_keeps_separate_results(self):
        result_ids = []
        download_names = []
        for color in ((0, 0, 255), (0, 255, 0)):
            image = np.full((32, 48, 3), color, dtype=np.uint8)
            success, encoded = cv2.imencode(".png", image)
            self.assertTrue(success)
            response = self.client.post(
                "/detect",
                data={"image": (io.BytesIO(encoded.tobytes()), "same-name.png")},
                content_type="multipart/form-data",
            )
            self.assertEqual(response.status_code, 200)
            download_path = re.search(
                rb'href="(/download/[^"]+)"', response.data
            ).group(1).decode()
            result_ids.append(download_path.rsplit("/", 1)[1])
            download = self.client.get(download_path)
            self.assertEqual(download.status_code, 200)
            content_disposition = download.headers["Content-Disposition"]
            download_names.append(
                content_disposition.split("filename=", 1)[1].strip('"')
            )
            download.close()

        self.assertNotEqual(result_ids[0], result_ids[1])
        self.assertEqual(len(set(download_names)), 2)
        self.assertTrue(
            all(name.startswith("same-name_ket_qua_") for name in download_names)
        )
        input_files = list(
            self.client.application.config["RESULTS_DIR"].glob("*_input.jpg")
        )
        result_files = list(
            self.client.application.config["RESULTS_DIR"].glob("*_result.jpg")
        )
        self.assertEqual(len(input_files), 2)
        self.assertEqual(len(result_files), 2)
        self.assertEqual(
            {path.stem.split("_")[0] for path in input_files},
            set(result_ids),
        )

    def test_missing_image_result_has_a_clear_not_found_message(self):
        missing_id = "00000000-0000-0000-0000-000000000001"

        download = self.client.get(f"/download/{missing_id}")
        history = self.client.get(f"/history/{missing_id}")

        self.assertEqual(download.status_code, 404)
        self.assertIn("Không tìm thấy ảnh kết quả".encode(), download.data)
        self.assertEqual(history.status_code, 404)
        self.assertIn("Không tìm thấy kết quả ảnh".encode(), history.data)

    def test_delete_history_supports_in_place_json_response(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        upload = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(encoded.tobytes()), "road.png")},
            content_type="multipart/form-data",
        )
        delete_path = re.search(
            rb'action="(/history/[^"]+/delete)"', upload.data
        ).group(1).decode()

        response = self.client.post(
            delete_path,
            headers={"Accept": "application/json"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"deleted": True})
        self.assertNotIn(b'class="history-row"', self.client.get("/").data)

    def test_accepts_tiff_even_when_filename_extension_is_unknown(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 48), (30, 100, 70)).save(stream, format="TIFF")
        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(stream.getvalue()), "camera.unknown")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="results"', response.data)

    def test_accepts_heic_even_when_filename_extension_is_unknown(self):
        heif = from_pillow(Image.new("RGB", (32, 48), (30, 100, 70)))
        stream = io.BytesIO()
        heif.save(stream, format="HEIF")
        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(stream.getvalue()), "phone.unknown")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'class="results"', response.data)

    def test_rejects_non_image_content_with_clear_message(self):
        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(b"not an image"), "notes.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Không đọc được nội dung ảnh".encode(), response.data)
        self.assertIn("HEIC/HEIF".encode(), response.data)

    def test_rejects_upload_over_file_size_limit(self):
        response = self.client.post(
            "/detect",
            data=b"x" * (12 * 1024 * 1024 + 1),
            content_type="application/octet-stream",
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("vượt quá giới hạn 12 MB".encode(), response.data)

    def test_rejects_decompression_bomb_warning_as_invalid_image(self):
        stream = io.BytesIO()
        Image.new("RGB", (32, 48), (30, 100, 70)).save(stream, format="PNG")

        with patch("web.app.Image.MAX_IMAGE_PIXELS", 10):
            response = self.client.post(
                "/detect",
                data={"image": (io.BytesIO(stream.getvalue()), "large.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Không đọc được nội dung ảnh".encode(), response.data)

    def test_model_load_failure_does_not_expose_internal_paths(self):
        root = Path(self.temp_dir.name)
        private_model_path = root / "internal" / "secret-helmet.pt"
        app = create_app(
            {
                "TESTING": True,
                "RESULTS_DIR": root / "model-error-results",
                "HELMET_MODEL": private_model_path,
                "PASSENGER_MODEL": root / "missing-passenger.pt",
                "CARGO_MODEL": root / "missing-cargo.pt",
            }
        )
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)

        with app.test_client() as client:
            response = client.post(
                "/detect",
                data={"image": (io.BytesIO(encoded.tobytes()), "road.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Không thể hoàn tất nhận diện ảnh".encode(), response.data)
        self.assertNotIn(str(root).encode(), response.data)
        self.assertNotIn(b"secret-helmet.pt", response.data)
        app.extensions["comparison_executor"].shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
