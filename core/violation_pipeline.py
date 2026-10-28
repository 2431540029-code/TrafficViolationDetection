from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from detection.cargo.classes import normalize_cargo_classes


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float
    class_name: str
    track_id: Optional[int] = None


@dataclass
class Vehicle:
    box: Box
    track_id: Optional[int]
    riders: int = 0
    trippling_confidence: Optional[float] = None
    no_helmet_confidence: Optional[float] = None
    cargo_confidence: Optional[float] = None


def model_classes(model):
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name).strip().lower()
                for class_id, name in names.items()}
    return {class_id: str(name).strip().lower()
            for class_id, name in enumerate(names)}


def read_detections(result, names):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = (
            int(value) for value in box.xyxy[0].tolist()
        )
        track_id = None
        if box.id is not None:
            track_id = int(box.id[0].item())
        detections.append(
            Detection(
                box=(x1, y1, x2, y2),
                confidence=float(box.conf[0].item()),
                class_name=names[class_id],
                track_id=track_id,
            )
        )
    return detections


def merge_overlapping_detections(detections, iou_threshold=0.5):
    kept = []
    for detection in sorted(
        detections, key=lambda item: item.confidence, reverse=True
    ):
        x1, y1, x2, y2 = detection.box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        duplicate = False
        for current in kept:
            if current.class_name != detection.class_name:
                continue
            cx1, cy1, cx2, cy2 = current.box
            intersection = max(0, min(x2, cx2) - max(x1, cx1)) * max(
                0, min(y2, cy2) - max(y1, cy1)
            )
            current_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
            union = area + current_area - intersection
            if union and intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def associate_with_vehicle(detection, vehicles):
    """Assign a head/person/cargo box to one nearby vehicle, at most once."""
    x1, y1, x2, y2 = detection.box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    best_index = None
    best_score = float("inf")

    for index, vehicle in enumerate(vehicles):
        vx1, vy1, vx2, vy2 = vehicle.box
        width = max(1, vx2 - vx1)
        height = max(1, vy2 - vy1)
        roi_x1 = vx1 - width * 0.35
        roi_x2 = vx2 + width * 0.35
        roi_y1 = vy1 - height * 1.5
        roi_y2 = vy2 + height * 0.2

        if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
            continue

        normalized_x = abs(center_x - (vx1 + vx2) / 2) / width
        normalized_y = abs(center_y - (vy1 + vy2) / 2) / height
        score = normalized_x + 0.25 * normalized_y
        if score < best_score:
            best_index = index
            best_score = score

    return best_index


def assign_cargo_to_vehicles(detections, vehicles):
    unassigned = []
    for detection in detections:
        index = associate_with_vehicle(detection, vehicles)
        if index is None:
            unassigned.append(detection)
            continue
        current = vehicles[index].cargo_confidence
        if current is None or detection.confidence > current:
            vehicles[index].cargo_confidence = detection.confidence
    return unassigned


def is_passenger_violation(vehicle):
    return vehicle.riders > 2 or vehicle.trippling_confidence is not None


class ViolationPipeline:
    IMAGE_TILE_SIZE = 768
    IMAGE_TILE_MIN_DIMENSION = IMAGE_TILE_SIZE * 2
    IMAGE_TILE_OVERLAP = 0.2

    def __init__(
        self,
        helmet_model_path: Path,
        passenger_model_path: Path,
        cargo_model_path: Optional[Path] = None,
        confidence: float = 0.35,
        image_passenger_confidence: float = 0.2,
        no_helmet_confidence: float = 0.48,
    ):
        for model_path in (helmet_model_path, passenger_model_path):
            if not model_path.is_file():
                raise FileNotFoundError(f"Khong tim thay model: {model_path}")

        self.helmet_model = YOLO(str(helmet_model_path))
        self.passenger_model = YOLO(str(passenger_model_path))
        self.helmet_names = model_classes(self.helmet_model)
        self.passenger_names = model_classes(self.passenger_model)
        self.confidence = confidence
        self.image_passenger_confidence = image_passenger_confidence
        self.no_helmet_confidence = no_helmet_confidence
        self.cargo_model = None
        self.cargo_names = {}
        self.cargo_class_ids = []

        self._require_classes(
            self.helmet_names, {"helmet", "no_helmet"}, "helmet"
        )
        self._require_classes(
            self.passenger_names, {"bike", "rider", "trippling"}, "passenger"
        )

        if cargo_model_path is not None and cargo_model_path.is_file():
            self.cargo_model = YOLO(str(cargo_model_path))
            self.cargo_class_ids, self.cargo_names = normalize_cargo_classes(
                self.cargo_model.names
            )
            if "protrusion" in {
                str(name).strip().lower()
                for name in model_classes(self.cargo_model).values()
            }:
                print(
                    "Thong tin: class protrusion cua model cargo "
                    "duoc hien thi nhu bulky_cargo."
                )
        else:
            print(
                "CANH BAO: Chua co checkpoint bulky_cargo; "
                "nhan dien hang cong kenh dang tat."
            )

    @staticmethod
    def _require_classes(names, required, model_name):
        missing = required.difference(names.values())
        if missing:
            raise ValueError(
                f"Model {model_name} thieu class {sorted(missing)}; "
                f"cac class hien co: {names}"
            )

    @staticmethod
    def _class_ids(names, target_names):
        return [
            class_id for class_id, name in names.items()
            if name in target_names
        ]

    def process(self, frame):
        annotated, events, _ = self.process_with_summary(frame)
        return annotated, events

    @staticmethod
    def _tile_starts(length, tile_size, overlap):
        if length <= tile_size:
            return [0]
        last_start = length - tile_size
        max_step = tile_size * (1 - overlap)
        step_count = max(1, round(last_start / max_step))
        while last_start / step_count > max_step * 1.01:
            step_count += 1
        return [
            round(index * last_start / step_count)
            for index in range(step_count + 1)
        ]

    def _image_regions(self, frame):
        height, width = frame.shape[:2]
        tile_size = self.IMAGE_TILE_SIZE
        regions = [(0, 0, frame)]
        if max(width, height) <= self.IMAGE_TILE_MIN_DIMENSION:
            return regions
        regions.extend(
            (
                x,
                y,
                frame[y:min(y + tile_size, height), x:min(x + tile_size, width)],
            )
            for y in self._tile_starts(
                height, tile_size, self.IMAGE_TILE_OVERLAP
            )
            for x in self._tile_starts(
                width, tile_size, self.IMAGE_TILE_OVERLAP
            )
        )
        return regions

    def _predict_image_detections(
        self, model, frame, names, target_names, confidence=None
    ):
        detections = []
        class_ids = self._class_ids(names, target_names)
        for x, y, region in self._image_regions(frame):
            result = model.predict(
                region,
                conf=(
                    self.confidence if confidence is None else confidence
                ),
                classes=class_ids,
                imgsz=640,
                verbose=False,
            )[0]
            detections.extend(
                Detection(
                    box=(
                        detection.box[0] + x,
                        detection.box[1] + y,
                        detection.box[2] + x,
                        detection.box[3] + y,
                    ),
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                )
                for detection in read_detections(result, names)
            )
        return merge_overlapping_detections(detections)

    def _filter_helmet_detections(self, detections):
        return [
            detection
            for detection in detections
            if detection.confidence
            >= (
                self.no_helmet_confidence
                if detection.class_name == "no_helmet"
                else self.confidence
            )
        ]

    def process_with_summary(self, frame, track=True):
        if track:
            passenger_result = self.passenger_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                classes=self._class_ids(
                    self.passenger_names, {"bike", "rider", "trippling"}
                ),
                verbose=False,
            )[0]
            passenger_detections = read_detections(
                passenger_result, self.passenger_names
            )
            helmet_result = self.helmet_model.predict(
                frame,
                conf=min(self.confidence, self.no_helmet_confidence),
                classes=self._class_ids(
                    self.helmet_names, {"helmet", "no_helmet"}
                ),
                verbose=False,
            )[0]
            helmet_detections = read_detections(
                helmet_result, self.helmet_names
            )
            helmet_detections = self._filter_helmet_detections(
                helmet_detections
            )
        else:
            passenger_detections = self._predict_image_detections(
                self.passenger_model,
                frame,
                self.passenger_names,
                {"bike", "rider", "trippling"},
                confidence=self.image_passenger_confidence,
            )
            helmet_detections = self._filter_helmet_detections(
                self._predict_image_detections(
                    self.helmet_model,
                    frame,
                    self.helmet_names,
                    {"helmet", "no_helmet"},
                    confidence=min(
                        self.confidence, self.no_helmet_confidence
                    ),
                )
            )
        cargo_detections = []
        if self.cargo_model is not None:
            cargo_result = self.cargo_model.predict(
                frame,
                conf=self.confidence,
                classes=self.cargo_class_ids,
                verbose=False,
            )[0]
            cargo_detections = read_detections(
                cargo_result, self.cargo_names
            )

        vehicles = []
        for detection in passenger_detections:
            if detection.class_name == "bike":
                vehicles.append(Vehicle(detection.box, detection.track_id))

        for detection in passenger_detections:
            if detection.class_name == "rider":
                index = associate_with_vehicle(detection, vehicles)
                if index is not None:
                    vehicles[index].riders += 1
            elif detection.class_name == "trippling":
                index = associate_with_vehicle(detection, vehicles)
                if index is not None:
                    current = vehicles[index].trippling_confidence
                    if current is None or detection.confidence > current:
                        vehicles[index].trippling_confidence = detection.confidence

        associated_no_helmet_detections = set()
        for detection in helmet_detections:
            if detection.class_name == "no_helmet":
                index = associate_with_vehicle(detection, vehicles)
                if index is not None:
                    associated_no_helmet_detections.add(id(detection))
                    current = vehicles[index].no_helmet_confidence
                    if current is None or detection.confidence > current:
                        vehicles[index].no_helmet_confidence = detection.confidence

        unassigned_cargo = assign_cargo_to_vehicles(cargo_detections, vehicles)
        unassigned_cargo_ids = {id(detection) for detection in unassigned_cargo}
        # Cargo is a violation only when associated with a detected motorcycle.
        cargo_detections = [
            detection
            for detection in cargo_detections
            if id(detection) not in unassigned_cargo_ids
        ]

        annotated = frame.copy()
        events = []
        for detection in helmet_detections:
            self._draw_detection(
                annotated,
                detection,
                confirmed=(
                    detection.class_name != "no_helmet"
                    or id(detection) in associated_no_helmet_detections
                ),
            )
        for detection in cargo_detections:
            self._draw_detection(annotated, detection)

        for vehicle in vehicles:
            violations = []
            if vehicle.no_helmet_confidence is not None:
                violations.append("KHONG DOI MU")
            if is_passenger_violation(vehicle):
                violations.append("CHO QUA SO NGUOI")
            if vehicle.cargo_confidence is not None:
                violations.append("HANG CONG KENH")

            color = (0, 0, 255) if violations else (0, 200, 0)
            x1, y1, x2, y2 = vehicle.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            identity = (
                f" ID:{vehicle.track_id}" if vehicle.track_id is not None else ""
            )
            label = (
                f"{' | '.join(violations)}{identity}"
                if violations
                else f"XE{identity} | {vehicle.riders} nguoi"
            )
            cv2.putText(
                annotated,
                label,
                (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            for violation, score in (
                ("khong_doi_mu", vehicle.no_helmet_confidence),
                ("cho_qua_so_nguoi", vehicle.trippling_confidence),
                ("hang_cong_kenh", vehicle.cargo_confidence),
            ):
                if score is not None:
                    events.append(
                        {
                            "track_id": vehicle.track_id,
                            "violation": violation,
                            "passenger_count": vehicle.riders,
                            "confidence": f"{score:.4f}",
                        }
                    )
            if is_passenger_violation(vehicle) and vehicle.trippling_confidence is None:
                events.append(
                    {
                        "track_id": vehicle.track_id,
                        "violation": "cho_qua_so_nguoi",
                        "passenger_count": vehicle.riders,
                        "confidence": "",
                    }
                )

        if self.cargo_model is None:
            status = "HANG CONG KENH: CHUA CO MODEL"
            color = (0, 200, 255)
        else:
            status = "HANG CONG KENH: DA BAT"
            color = (255, 255, 255)
        cv2.putText(
            annotated,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        summary = {
            "helmet_on": sum(
                detection.class_name == "helmet"
                for detection in helmet_detections
            ),
            "no_helmet": sum(
                vehicle.no_helmet_confidence is not None
                for vehicle in vehicles
            ),
            "motorcycles": len(vehicles),
            "riders": sum(vehicle.riders for vehicle in vehicles),
            "passenger_violations": sum(
                is_passenger_violation(vehicle) for vehicle in vehicles
            ),
            "cargo_detections": len(cargo_detections),
            "cargo_enabled": self.cargo_model is not None,
            "violations": events,
        }
        return annotated, events, summary

    @staticmethod
    def _draw_detection(frame, detection, confirmed=True):
        x1, y1, x2, y2 = detection.box
        color = (
            (0, 165, 255)
            if detection.class_name == "no_helmet" and not confirmed
            else (0, 0, 255)
            if detection.class_name in {"no_helmet", "bulky_cargo"}
            else (255, 180, 0)
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            (
                f"{detection.class_name}? {detection.confidence:.2f}"
                if detection.class_name == "no_helmet" and not confirmed
                else f"{detection.class_name} {detection.confidence:.2f}"
            ),
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
