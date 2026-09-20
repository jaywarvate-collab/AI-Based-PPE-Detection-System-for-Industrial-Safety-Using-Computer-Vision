from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import csv
import os
import threading
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
SOURCE_DIR = os.path.join(BASE_DIR, "source_files")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
LOG_FILE = os.path.join(BASE_DIR, "violations.csv")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

model = YOLO(MODEL_PATH)

# Dataset classes used by the supplied best.pt model.
VIOLATION_CLASSES = {
    "NO-Hardhat": "No Hardhat",
    "NO-Mask": "No Mask",
    "NO-Safety Vest": "No Safety Vest",
}
POSITIVE_PPE = {"Hardhat", "Mask", "Safety Vest"}
PERSON_CLASS = "Person"

state_lock = threading.Lock()
state = {
    "workers": 0,
    "safe": 0,
    "violations": 0,
    "compliance": 0.0,
    "mode": "camera",
    "source_name": "Live Camera",
    "image_url": None,
    "frame_ready": False,
}
camera = None
camera_lock = threading.Lock()
last_logged = {}


def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Date", "Time", "Worker ID", "Violation", "Source"])


ensure_log()


def class_name(cls_id):
    return model.names[int(cls_id)]


def center(box):
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_expanded_box(point, box, expand_x=0.12, expand_top=0.30, expand_bottom=0.08):
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    px, py = point
    return (
        x1 - expand_x * w <= px <= x2 + expand_x * w
        and y1 - expand_top * h <= py <= y2 + expand_bottom * h
    )


def associate_ppe_to_workers(result):
    """
    Associate PPE/violation detections with the nearest person whose expanded
    bounding box contains the PPE center. This prevents a violation detected
    somewhere in the frame from being assigned to every worker.
    """
    boxes = result.boxes
    detections = []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        name = class_name(cls_id)
        conf = float(boxes.conf[i])
        xyxy = boxes.xyxy[i].cpu().tolist()
        track_id = None
        if boxes.id is not None:
            try:
                track_id = int(boxes.id[i])
            except Exception:
                track_id = None
        detections.append({
            "name": name,
            "conf": conf,
            "box": xyxy,
            "track_id": track_id,
        })

    persons = [d for d in detections if d["name"] == PERSON_CLASS]
    ppe = [d for d in detections if d["name"] in VIOLATION_CLASSES or d["name"] in POSITIVE_PPE]

    workers = []
    for idx, person in enumerate(persons):
        worker = {
            "index": idx + 1,
            "track_id": person["track_id"],
            "box": person["box"],
            "violations": [],
            "ppe": [],
        }
        workers.append(worker)

    # Assign each PPE detection to at most one worker.
    for item in ppe:
        c = center(item["box"])
        candidates = []
        for w in workers:
            if point_in_expanded_box(c, w["box"]):
                pc = center(w["box"])
                # normalized distance makes the association scale-aware
                dx = (c[0] - pc[0]) / max(1.0, w["box"][2] - w["box"][0])
                dy = (c[1] - pc[1]) / max(1.0, w["box"][3] - w["box"][1])
                candidates.append((dx * dx + dy * dy, w))

        if candidates:
            _, target = min(candidates, key=lambda x: x[0])
            target["ppe"].append(item["name"])
            if item["name"] in VIOLATION_CLASSES:
                target["violations"].append(item["name"])

    # If a worker has multiple duplicate detections of the same class, keep one.
    for w in workers:
        w["violations"] = sorted(set(w["violations"]))
        w["ppe"] = sorted(set(w["ppe"]))

    return workers


def update_stats(workers):
    total = len(workers)
    unsafe = sum(1 for w in workers if w["violations"])
    safe = total - unsafe
    compliance = round((safe / total) * 100, 1) if total else 0.0

    with state_lock:
        state["workers"] = total
        state["safe"] = safe
        state["violations"] = unsafe
        state["compliance"] = compliance
        state["frame_ready"] = True


def log_worker_violations(workers, source_name):
    now = datetime.now()
    for w in workers:
        worker_key = w["track_id"] if w["track_id"] is not None else f"person-{w['index']}"
        for violation in w["violations"]:
            key = (source_name, worker_key, violation)
            previous = last_logged.get(key)
            if previous is None or (now - previous).total_seconds() >= 5:
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        now.strftime("%d-%m-%Y"),
                        now.strftime("%H:%M:%S"),
                        str(worker_key),
                        violation,
                        source_name,
                    ])
                last_logged[key] = now


def annotate(result):
    frame = result.plot()

    workers = associate_ppe_to_workers(result)
    update_stats(workers)
    return frame, workers


def add_status_overlay(frame, workers):
    if not workers:
        text = "NO WORKERS DETECTED"
        color = (0, 215, 255)
    elif any(w["violations"] for w in workers):
        text = "SAFETY VIOLATION"
        color = (0, 0, 255)
    else:
        text = "PPE STATUS: SAFE"
        color = (0, 180, 0)

    cv2.rectangle(frame, (10, 10), (420, 58), (20, 20, 20), -1)
    cv2.putText(frame, text, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Show per-worker violation information near the top-left.
    y = 88
    for w in workers:
        if w["violations"]:
            label = f"Worker {w['track_id'] if w['track_id'] is not None else w['index']}: " + \
                    ", ".join(VIOLATION_CLASSES[v] for v in w["violations"])
            cv2.putText(frame, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
            y += 24

    return frame


def process_frame(frame, source_name, use_tracking=True, confidence=0.25):
    if use_tracking:
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            iou=0.50,
            verbose=False,
        )
    else:
        results = model.predict(
            source=frame,
            conf=confidence,
            iou=0.50,
            verbose=False,
        )

    result = results[0]
    annotated, workers = annotate(result)
    log_worker_violations(workers, source_name)
    annotated = add_status_overlay(annotated, workers)
    return annotated


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return camera


def generate_frames():
    mode = state["mode"]
    source_name = state["source_name"]

    if mode == "image":
        image_url = state["image_url"]
        if image_url:
            path = os.path.join(BASE_DIR, image_url.lstrip("/").replace("/", os.sep))
            frame = cv2.imread(path)
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        return

    if mode == "video":
        video_path = state.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            annotated = process_frame(frame, source_name, use_tracking=True)
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.01)
        cap.release()
        return

    # Live camera
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        annotated = process_frame(frame, "Live Camera", use_tracking=True)
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def get_violations(limit=100):
    ensure_log()
    rows = []
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return list(reversed(rows))[:limit]


def available_sources():
    allowed = {".jpg", ".jpeg", ".png", ".jfif", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = []
    if os.path.isdir(SOURCE_DIR):
        for name in sorted(os.listdir(SOURCE_DIR)):
            path = os.path.join(SOURCE_DIR, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in allowed:
                files.append(name)
    return files


@app.route("/")
def home():
    return render_template(
        "index.html",
        violations=get_violations(),
        sources=available_sources(),
    )


@app.route("/stats")
def stats():
    with state_lock:
        data = dict(state)
    data["recent_violations"] = get_violations(20)
    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )


@app.route("/set_camera", methods=["POST"])
def set_camera():
    with state_lock:
        state["mode"] = "camera"
        state["source_name"] = "Live Camera"
        state["image_url"] = None
        state.pop("video_path", None)
    return jsonify({"ok": True, "mode": "camera"})


@app.route("/use_source", methods=["POST"])
def use_source():
    filename = secure_filename(request.form.get("filename", ""))
    path = os.path.join(SOURCE_DIR, filename)

    if not filename or not os.path.isfile(path):
        return jsonify({"ok": False, "error": "Source file not found."}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".jfif"}:
        # Process image once and save the annotated result.
        frame = cv2.imread(path)
        if frame is None:
            return jsonify({"ok": False, "error": "Could not read image."}), 400

        annotated = process_frame(frame, filename, use_tracking=False)
        out_name = f"processed_{int(time.time())}_{filename.rsplit('.', 1)[0]}.jpg"
        out_path = os.path.join(RESULT_DIR, out_name)
        cv2.imwrite(out_path, annotated)

        with state_lock:
            state["mode"] = "image"
            state["source_name"] = filename
            state["image_url"] = "/static/results/" + out_name
            state.pop("video_path", None)

        return jsonify({"ok": True, "mode": "image", "image_url": state["image_url"]})

    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        with state_lock:
            state["mode"] = "video"
            state["source_name"] = filename
            state["video_path"] = path
            state["image_url"] = None
        return jsonify({"ok": True, "mode": "video"})

    return jsonify({"ok": False, "error": "Unsupported file type."}), 400


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file selected."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".jpg", ".jpeg", ".png", ".jfif", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed:
        return jsonify({"ok": False, "error": "Use JPG/PNG/JFIF or MP4/AVI/MOV/MKV/WEBM."}), 400

    unique_name = f"{int(time.time())}_{filename}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(path)

    if ext in {".jpg", ".jpeg", ".png", ".jfif"}:
        frame = cv2.imread(path)
        if frame is None:
            return jsonify({"ok": False, "error": "Uploaded image could not be read."}), 400

        annotated = process_frame(frame, filename, use_tracking=False)
        out_name = f"processed_{unique_name.rsplit('.', 1)[0]}.jpg"
        out_path = os.path.join(RESULT_DIR, out_name)
        cv2.imwrite(out_path, annotated)

        with state_lock:
            state["mode"] = "image"
            state["source_name"] = filename
            state["image_url"] = "/static/results/" + out_name
            state.pop("video_path", None)

        return jsonify({"ok": True, "mode": "image", "image_url": state["image_url"]})

    with state_lock:
        state["mode"] = "video"
        state["source_name"] = filename
        state["video_path"] = path
        state["image_url"] = None

    return jsonify({"ok": True, "mode": "video"})



@app.route("/process_webcam_frame", methods=["POST"])
def process_webcam_frame():
    """Process one frame captured by the user's browser webcam."""
    if "frame" not in request.files:
        return jsonify({"ok": False, "error": "No webcam frame received."}), 400

    file = request.files["frame"]
    data = file.read()
    if not data:
        return jsonify({"ok": False, "error": "Empty webcam frame."}), 400

    array = __import__("numpy").frombuffer(data, dtype=__import__("numpy").uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode webcam frame."}), 400

    try:
       annotated = process_frame(
    frame,
    "Browser Webcam",
    use_tracking=True,
    confidence=0.15
)
        ok, buffer = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        )
        if not ok:
            return jsonify({"ok": False, "error": "Could not encode detection result."}), 500

        return Response(buffer.tobytes(), mimetype="image/jpeg", headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

@app.route("/reset_log", methods=["POST"])
def reset_log():
    ensure_log()
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Date", "Time", "Worker ID", "Violation", "Source"])
    last_logged.clear()
    return jsonify({"ok": True})


@app.route("/download_log")
def download_log():
    ensure_log()
    return send_from_directory(BASE_DIR, "violations.csv", as_attachment=True)


if __name__ == "__main__":
    # use_reloader=False prevents Flask debug mode from opening the webcam twice.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True, threaded=True, use_reloader=False)
