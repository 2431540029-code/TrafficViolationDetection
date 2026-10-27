import io
import json
import logging
import os
import sqlite3
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from flask import (
    Flask,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from core.violation_pipeline import ViolationPipeline
from detection.cargo.paths import latest_cargo_checkpoint
from web.model_comparison import (
    TASKS,
    evaluate_model,
    get_model_catalog,
    initialize_evaluation_history,
    load_evaluation_history,
    model_is_ready,
    model_weight_path,
    save_evaluation_results,
)


BASE_DIR = Path(__file__).resolve().parents[1]
register_heif_opener()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
DEFAULT_HELMET_MODEL = (
    BASE_DIR / "runs" / "detect" / "helmet_model-2" / "weights" / "best.pt"
)
DEFAULT_PASSENGER_MODEL = (
    BASE_DIR / "runs" / "detect" / "v02_trippling" / "weights" / "best.pt"
)
DEFAULT_CARGO_MODEL = latest_cargo_checkpoint(BASE_DIR)


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        RESULTS_DIR=Path(
            os.environ.get("TRAFFIC_WEB_RESULTS", BASE_DIR / "web_results")
        ),
        HISTORY_DB=None,
        HELMET_MODEL=DEFAULT_HELMET_MODEL,
        PASSENGER_MODEL=DEFAULT_PASSENGER_MODEL,
        CARGO_MODEL=DEFAULT_CARGO_MODEL,
        CONFIDENCE=0.35,
        IMAGE_PASSENGER_CONFIDENCE=0.3,
        PIPELINE=None,
        COMPARISON_RUNNER=None,
    )
    if test_config:
        app.config.update(test_config)
    for task_definition in TASKS.values():
        for model_definition in task_definition["models"].values():
            app.config.setdefault(
                model_definition["weight_config"],
                BASE_DIR / model_definition["weight_file"],
            )

    app.config["RESULTS_DIR"] = Path(app.config["RESULTS_DIR"])
    app.config["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)
    if app.config["HISTORY_DB"] is None:
        app.config["HISTORY_DB"] = app.config["RESULTS_DIR"] / "history.sqlite3"
    app.config["HISTORY_DB"] = Path(app.config["HISTORY_DB"])
    app.config["HISTORY_DB"].parent.mkdir(parents=True, exist_ok=True)
    initialize_history(app.config["HISTORY_DB"])
    initialize_evaluation_history(app.config["HISTORY_DB"])
    app.extensions["detection_lock"] = threading.Lock()
    app.extensions["comparison_jobs_lock"] = threading.Lock()
    app.extensions["comparison_jobs"] = {}
    app.extensions["comparison_active_job"] = None
    app.extensions["comparison_executor"] = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="ai-comparison"
    )

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            models={
                "helmet": Path(app.config["HELMET_MODEL"]).is_file(),
                "passenger": Path(app.config["PASSENGER_MODEL"]).is_file(),
                "cargo": Path(app.config["CARGO_MODEL"]).is_file(),
            },
            history=load_history(app.config["HISTORY_DB"]),
        )

    @app.get("/ai-comparison")
    def ai_comparison():
        return render_template(
            "ai_comparison.html",
            catalog=get_model_catalog(app.config),
        )

    @app.post("/api/ai-comparison/evaluate")
    def start_ai_comparison():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Yêu cầu phải là JSON hợp lệ."}), 400

        task_id = payload.get("task")
        model_ids = payload.get("model_ids")
        split = payload.get("split")
        confidence = payload.get("confidence")
        repeats = payload.get("repeats")
        if not isinstance(task_id, str) or task_id not in TASKS:
            return jsonify({"error": "Bài toán được chọn không hợp lệ."}), 400
        if (
            not isinstance(model_ids, list)
            or not model_ids
            or len(model_ids) > 3
            or any(not isinstance(model_id, str) for model_id in model_ids)
            or len(set(model_ids)) != len(model_ids)
        ):
            return jsonify({"error": "Hãy chọn từ 1 đến 3 mô hình khác nhau."}), 400
        definition = TASKS[task_id]
        if any(model_id not in definition["models"] for model_id in model_ids):
            return jsonify({
                "error": "Mô hình không tương thích với bài toán đã chọn."
            }), 400
        if not isinstance(split, str) or split not in {"val", "test"}:
            return jsonify({"error": "Tập đánh giá không hợp lệ."}), 400
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.01 <= confidence <= 0.99
        ):
            return jsonify({
                "error": "Confidence phải nằm trong khoảng 0.01 đến 0.99."
            }), 400
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
            return jsonify({"error": "Số lần đo phải từ 1 đến 5."}), 400

        missing_models = [
            model_id
            for model_id in model_ids
            if not model_is_ready(app.config, task_id, model_id)
        ]
        if missing_models:
            return jsonify({
                "error": (
                    "Chưa có checkpoint đã huấn luyện cho: "
                    + ", ".join(missing_models)
                    + ". Hãy huấn luyện các mô hình này trước khi đánh giá."
                )
            }), 409

        with app.extensions["comparison_jobs_lock"]:
            active_job = app.extensions["comparison_active_job"]
            if active_job is not None:
                return jsonify({
                    "error": "Một lượt đánh giá khác đang chạy. Vui lòng chờ hoàn tất."
                }), 409

            job_id = str(uuid.uuid4())
            job = {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "results": [],
                "error": None,
            }
            app.extensions["comparison_jobs"][job_id] = job
            app.extensions["comparison_active_job"] = job_id
            app.extensions["comparison_executor"].submit(
                run_ai_comparison_job,
                app,
                job_id,
                task_id,
                model_ids,
                split,
                float(confidence),
                repeats,
            )
        return jsonify(job), 202

    @app.get("/api/ai-comparison/jobs/<uuid:job_id>")
    def get_ai_comparison_job(job_id):
        with app.extensions["comparison_jobs_lock"]:
            job = app.extensions["comparison_jobs"].get(str(job_id))
            if job is None:
                return jsonify({"error": "Không tìm thấy lượt đánh giá."}), 404
            return jsonify(dict(job))

    @app.get("/api/ai-comparison/history")
    def ai_comparison_history():
        task_id = request.args.get("task")
        if task_id and task_id not in TASKS:
            return jsonify({"error": "Bài toán được chọn không hợp lệ."}), 400
        return jsonify({
            "results": load_evaluation_history(
                app.config["HISTORY_DB"], task=task_id
            )
        })

    @app.post("/detect")
    def detect():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template(
                "index.html",
                error="Bạn hãy chọn một ảnh trước khi nhận diện.",
                models=model_status(app),
                history=load_history(app.config["HISTORY_DB"]),
            ), 400

        encoded = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(encoded) > MAX_UPLOAD_BYTES:
            return render_template(
                "index.html",
                error="Ảnh vượt quá giới hạn 12 MB.",
                models=model_status(app),
                history=load_history(app.config["HISTORY_DB"]),
            ), 413
        if not encoded:
            return render_template(
                "index.html",
                error="Tệp ảnh trống.",
                models=model_status(app),
                history=load_history(app.config["HISTORY_DB"]),
            ), 400

        try:
            image = decode_uploaded_image(encoded)
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            return render_template(
                "index.html",
                error=(
                    "Không đọc được nội dung ảnh. Hãy chọn ảnh raster hợp lệ "
                    "như JPEG, PNG, GIF, TIFF, BMP hoặc WEBP."
                ),
                models=model_status(app),
                history=load_history(app.config["HISTORY_DB"]),
            ), 400

        result_id = str(uuid.uuid4())
        safe_name = uploaded.filename.replace("\\", "/").rsplit("/", 1)[-1]
        original_name = safe_name[:255] or "anh-khong-ten"
        saved_paths = []
        try:
            with app.extensions["detection_lock"]:
                pipeline = get_pipeline(app)
                annotated, _, summary = pipeline.process_with_summary(
                    image, track=False
                )
            results_dir = app.config["RESULTS_DIR"]
            input_name = f"{result_id}_input.jpg"
            output_name = f"{result_id}_result.jpg"

            input_path = results_dir / input_name
            output_path = results_dir / output_name
            if not cv2.imwrite(
                str(input_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise OSError("Không thể lưu ảnh đầu vào trên máy chủ.")
            saved_paths.append(input_path)
            if not cv2.imwrite(
                str(output_path),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise OSError("Không thể lưu ảnh kết quả trên máy chủ.")
            saved_paths.append(output_path)
            save_history(
                app.config["HISTORY_DB"],
                result_id=result_id,
                original_name=original_name,
                input_name=input_name,
                output_name=output_name,
                summary=summary,
            )
        except Exception as error:
            for saved_path in saved_paths:
                saved_path.unlink(missing_ok=True)
            current_app.logger.exception("Image detection failed")
            return render_template(
                "index.html",
                error=f"Nhận diện thất bại: {error}",
                models=model_status(app),
                history=load_history(app.config["HISTORY_DB"]),
            ), 500

        result = {
            "id": result_id,
            "original_name": original_name,
            "input_url": url_for("media", result_id=result_id, kind="input"),
            "output_url": url_for("media", result_id=result_id, kind="result"),
            "download_url": url_for("download", result_id=result_id),
            "summary": summary,
        }
        return render_template(
            "index.html",
            result=result,
            models=model_status(app),
            history=load_history(app.config["HISTORY_DB"]),
        )

    @app.get("/history/<uuid:result_id>")
    def history_detail(result_id):
        record = get_history_record(app.config["HISTORY_DB"], str(result_id))
        if record is None:
            abort(404)
        return render_template(
            "index.html",
            result={
                "id": record["id"],
                "original_name": record["original_name"],
                "input_url": url_for(
                    "media", result_id=record["id"], kind="input"
                ),
                "output_url": url_for(
                    "media", result_id=record["id"], kind="result"
                ),
                "download_url": url_for("download", result_id=record["id"]),
                "summary": json.loads(record["summary_json"]),
            },
            models=model_status(app),
            history=load_history(app.config["HISTORY_DB"]),
        )

    @app.post("/history/<uuid:result_id>/delete")
    def delete_history_entry(result_id):
        record = get_history_record(app.config["HISTORY_DB"], str(result_id))
        if record is None:
            abort(404)

        with closing(sqlite3.connect(app.config["HISTORY_DB"])) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM analysis_history WHERE id = ?",
                    (str(result_id),),
                )
        for filename in (record["input_name"], record["output_name"]):
            (app.config["RESULTS_DIR"] / filename).unlink(missing_ok=True)
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"deleted": True})
        return redirect(url_for("index"))

    @app.get("/media/<uuid:result_id>/<kind>")
    def media(result_id, kind):
        if kind not in {"input", "result"}:
            abort(404)
        return send_from_directory(
            app.config["RESULTS_DIR"],
            f"{result_id}_{kind}.jpg",
            mimetype="image/jpeg",
        )

    @app.get("/download/<uuid:result_id>")
    def download(result_id):
        return send_from_directory(
            app.config["RESULTS_DIR"],
            f"{result_id}_result.jpg",
            as_attachment=True,
            download_name="ket_qua_nhan_dien.jpg",
        )

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        current_app.logger.warning("Rejected oversized upload: %s", _error)
        return render_template(
            "index.html",
            error="Ảnh vượt quá giới hạn 12 MB.",
            models=model_status(app),
            history=load_history(app.config["HISTORY_DB"]),
        ), 413

    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    return app


def run_ai_comparison_job(
    app, job_id, task_id, model_ids, split, confidence, repeats
):
    with app.app_context():
        jobs = app.extensions["comparison_jobs"]
        jobs_lock = app.extensions["comparison_jobs_lock"]
        with jobs_lock:
            jobs[job_id]["status"] = "running"

        try:
            runner = app.config.get("COMPARISON_RUNNER") or evaluate_model
            results = []
            with app.extensions["detection_lock"]:
                for index, model_id in enumerate(model_ids):
                    weight_path = model_weight_path(
                        app.config, task_id, model_id
                    )
                    results.append(
                        runner(
                            weight_path,
                            task_id,
                            model_id,
                            split,
                            confidence,
                            repeats,
                            job_id,
                        )
                    )
                    with jobs_lock:
                        jobs[job_id]["progress"] = round(
                            (index + 1) / len(model_ids) * 100
                        )
            save_evaluation_results(
                app.config["HISTORY_DB"], job_id, results
            )
        except Exception:
            app.logger.exception("AI model comparison failed")
            with jobs_lock:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = (
                    "Đánh giá thất bại. Kiểm tra cấu hình dữ liệu và log máy chủ."
                )
        else:
            with jobs_lock:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["results"] = results
        finally:
            with jobs_lock:
                app.extensions["comparison_active_job"] = None


def get_pipeline(app):
    pipeline = app.config.get("PIPELINE")
    if pipeline is None:
        cargo_model = Path(app.config["CARGO_MODEL"])
        pipeline = ViolationPipeline(
            helmet_model_path=Path(app.config["HELMET_MODEL"]),
            passenger_model_path=Path(app.config["PASSENGER_MODEL"]),
            cargo_model_path=cargo_model if cargo_model.is_file() else None,
            confidence=app.config["CONFIDENCE"],
            image_passenger_confidence=app.config[
                "IMAGE_PASSENGER_CONFIDENCE"
            ],
        )
        app.config["PIPELINE"] = pipeline
    return pipeline


def model_status(app):
    return {
        "helmet": Path(app.config["HELMET_MODEL"]).is_file(),
        "passenger": Path(app.config["PASSENGER_MODEL"]).is_file(),
        "cargo": Path(app.config["CARGO_MODEL"]).is_file(),
    }


def decode_uploaded_image(encoded):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(encoded)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            pixels = np.asarray(image)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


def initialize_history(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_name TEXT NOT NULL,
                    output_name TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )


def save_history(database_path, result_id, original_name, input_name, output_name, summary):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO analysis_history
                    (id, original_name, created_at, input_name, output_name, summary_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    original_name,
                    created_at,
                    input_name,
                    output_name,
                    json.dumps(summary, ensure_ascii=False),
                ),
            )


def load_history(database_path, limit=50):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        records = connection.execute(
            """
            SELECT id, original_name, created_at, summary_json
            FROM analysis_history
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []
    for record in records:
        summary = json.loads(record["summary_json"])
        history.append(
            {
                "id": record["id"],
                "original_name": record["original_name"],
                "created_at": record["created_at"],
                "summary": summary,
                "detail_url": url_for("history_detail", result_id=record["id"]),
            }
        )
    return history


def get_history_record(database_path, result_id):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        record = connection.execute(
            "SELECT * FROM analysis_history WHERE id = ?",
            (result_id,),
        ).fetchone()
    return dict(record) if record is not None else None


def main():
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(
        host=os.environ.get("TRAFFIC_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRAFFIC_WEB_PORT", "5000")),
        debug=False,
    )


if __name__ == "__main__":
    main()
