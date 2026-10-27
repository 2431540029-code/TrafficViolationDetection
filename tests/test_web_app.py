import io
import re
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from pillow_heif import from_pillow
from PIL import Image

from web.app import create_app


class FakePipeline:
    def process_with_summary(self, image, track=True):
        self.track_enabled = track
        summary = {
            "helmet_on": 1,
            "no_helmet": 1,
            "motorcycles": 1,
            "riders": 3,
            "passenger_violations": 1,
            "cargo_detections": 0,
            "cargo_enabled": False,
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
        self.assertIn("THIẾU MODEL".encode(), response.data)
        self.assertIn("multipart/form-data".encode(), response.data)
        self.assertIn(b'class="history-section"', response.data)
        self.assertIn(b'class="history-empty"', response.data)
        self.assertIn(b'id="violation-models-table"', response.data)
        self.assertNotIn(b'id="model-performance-table"', response.data)
        self.assertNotIn(b'id="model-versions-table"', response.data)
        self.assertNotIn("Bảng 1".encode(), response.data)
        self.assertNotIn("Bảng 2".encode(), response.data)
        self.assertNotIn("Bảng 3".encode(), response.data)
        self.assertIn("Mô hình AI theo từng loại vi phạm".encode(), response.data)
        self.assertNotIn("biển số".encode(), response.data.lower())

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

    def test_ai_comparison_rejects_missing_weights_and_invalid_model(self):
        missing_weights = self.client.post(
            "/api/ai-comparison/evaluate",
            json={
                "task": "cargo",
                "model_ids": ["cargo_yolo11n"],
                "split": "test",
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
                "split": "test",
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
                "dataset": "Roboflow Cargo v3",
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
                "split": "test",
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
        self.assertIn(b'class="notice"', response.data)
        self.assertIn(b'class="history-row"', response.data)
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


if __name__ == "__main__":
    unittest.main()
