from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

import cv2
import csv
import os
import threading
import time
import numpy as np

from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "source_files"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "violations.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# PPE CLASSES
# ============================================================

VIOLATION_CLASSES = {
    "NO-Hardhat": "No Hardhat",
    "NO-Mask": "No Mask",
    "NO-Safety Vest": "No Safety Vest",
}

POSITIVE_PPE = {
    "Hardhat",
    "Mask",
    "Safety Vest",
}

PERSON_CLASS = "Person"


# ============================================================
# GLOBAL STATE
# ============================================================

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
    "video_path": None,
}


# Local/server camera support.
# Browser webcam does NOT use this on Render.
camera = None
camera_lock = threading.Lock()

last_logged = {}


# ============================================================
# CSV LOG
# ============================================================

def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            csv.writer(f).writerow(
                [
                    "Date",
                    "Time",
                    "Worker ID",
                    "Violation",
                    "Source",
                ]
            )


ensure_log()


# ============================================================
# YOLO HELPERS
# ============================================================

def class_name(cls_id):
    return model.names[int(cls_id)]


def center(box):
    x1, y1, x2, y2 = map(float, box)

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def point_in_expanded_box(
    point,
    box,
    expand_x=0.12,
    expand_top=0.30,
    expand_bottom=0.08
):
    x1, y1, x2, y2 = map(float, box)

    width = x2 - x1
    height = y2 - y1

    px, py = point

    return (
        x1 - expand_x * width
        <= px
        <= x2 + expand_x * width
        and
        y1 - expand_top * height
        <= py
        <= y2 + expand_bottom * height
    )


# ============================================================
# ASSOCIATE PPE WITH WORKERS
# ============================================================

def associate_ppe_to_workers(result):
    """
    Associates PPE and PPE violations with the nearest detected
    person whose expanded bounding box contains the PPE center.
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

        detections.append(
            {
                "name": name,
                "conf": conf,
                "box": xyxy,
                "track_id": track_id,
            }
        )

    # --------------------------------------------------------
    # Find people
    # --------------------------------------------------------

    persons = [
        d
        for d in detections
        if d["name"] == PERSON_CLASS
    ]

    # --------------------------------------------------------
    # Find PPE / violations
    # --------------------------------------------------------

    ppe = [
        d
        for d in detections
        if (
            d["name"] in VIOLATION_CLASSES
            or
            d["name"] in POSITIVE_PPE
        )
    ]

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

    # --------------------------------------------------------
    # Assign PPE to nearest worker
    # --------------------------------------------------------

    for item in ppe:

        item_center = center(item["box"])

        candidates = []

        for worker in workers:

            if point_in_expanded_box(
                item_center,
                worker["box"]
            ):

                person_center = center(
                    worker["box"]
                )

                worker_width = max(
                    1.0,
                    worker["box"][2]
                    - worker["box"][0]
                )

                worker_height = max(
                    1.0,
                    worker["box"][3]
                    - worker["box"][1]
                )

                dx = (
                    item_center[0]
                    - person_center[0]
                ) / worker_width

                dy = (
                    item_center[1]
                    - person_center[1]
                ) / worker_height

                distance = (
                    dx * dx
                    +
                    dy * dy
                )

                candidates.append(
                    (
                        distance,
                        worker
                    )
                )

        if candidates:

            _, target = min(
                candidates,
                key=lambda x: x[0]
            )

            target["ppe"].append(
                item["name"]
            )

            if item["name"] in VIOLATION_CLASSES:
                target["violations"].append(
                    item["name"]
                )

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    for worker in workers:

        worker["violations"] = sorted(
            set(worker["violations"])
        )

        worker["ppe"] = sorted(
            set(worker["ppe"])
        )

    return workers


# ============================================================
# UPDATE DASHBOARD STATISTICS
# ============================================================

def update_stats(workers):

    total_workers = len(workers)

    unsafe_workers = sum(
        1
        for worker in workers
        if worker["violations"]
    )

    safe_workers = (
        total_workers
        - unsafe_workers
    )

    if total_workers:
        compliance = round(
            (safe_workers / total_workers) * 100,
            1
        )
    else:
        compliance = 0.0

    with state_lock:

        state["workers"] = total_workers

        state["safe"] = safe_workers

        state["violations"] = unsafe_workers

        state["compliance"] = compliance

        state["frame_ready"] = True


# ============================================================
# LOG VIOLATIONS
# ============================================================

def log_worker_violations(
    workers,
    source_name
):

    now = datetime.now()

    for worker in workers:

        if worker["track_id"] is not None:

            worker_key = worker["track_id"]

        else:

            worker_key = (
                f"person-{worker['index']}"
            )

        for violation in worker["violations"]:

            key = (
                source_name,
                worker_key,
                violation,
            )

            previous = last_logged.get(key)

            should_log = (
                previous is None
                or
                (
                    now - previous
                ).total_seconds() >= 5
            )

            if should_log:

                with open(
                    LOG_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as f:

                    csv.writer(f).writerow(
                        [
                            now.strftime(
                                "%d-%m-%Y"
                            ),
                            now.strftime(
                                "%H:%M:%S"
                            ),
                            str(worker_key),
                            violation,
                            source_name,
                        ]
                    )

                last_logged[key] = now


# ============================================================
# ANNOTATE RESULT
# ============================================================

def annotate(result):

    frame = result.plot()

    workers = associate_ppe_to_workers(
        result
    )

    update_stats(workers)

    return frame, workers


# ============================================================
# STATUS OVERLAY
# ============================================================

def add_status_overlay(
    frame,
    workers
):

    if not workers:

        text = "NO WORKERS DETECTED"

        color = (
            0,
            215,
            255
        )

    elif any(
        worker["violations"]
        for worker in workers
    ):

        text = "SAFETY VIOLATION"

        color = (
            0,
            0,
            255
        )

    else:

        text = "PPE STATUS: SAFE"

        color = (
            0,
            180,
            0
        )

    # Status background
    cv2.rectangle(
        frame,
        (10, 10),
        (420, 58),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        text,
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2
    )

    # Worker violation information
    y = 88

    for worker in workers:

        if worker["violations"]:

            worker_id = (
                worker["track_id"]
                if worker["track_id"] is not None
                else worker["index"]
            )

            labels = ", ".join(
                VIOLATION_CLASSES[v]
                for v in worker["violations"]
            )

            label = (
                f"Worker {worker_id}: "
                f"{labels}"
            )

            cv2.putText(
                frame,
                label,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2
            )

            y += 24

    return frame


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(
    frame,
    source_name,
    use_tracking=True,
    confidence=0.25
):

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

    annotated, workers = annotate(
        result
    )

    log_worker_violations(
        workers,
        source_name
    )

    annotated = add_status_overlay(
        annotated,
        workers
    )

    return annotated


# ============================================================
# LOCAL CAMERA SUPPORT
# ============================================================

def get_camera():

    global camera

    with camera_lock:

        if (
            camera is None
            or not camera.isOpened()
        ):

            camera = cv2.VideoCapture(0)

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

        return camera


# ============================================================
# VIDEO STREAM GENERATOR
# ============================================================

def generate_frames():

    with state_lock:
        mode = state["mode"]
        source_name = state["source_name"]
        image_url = state["image_url"]
        video_path = state.get("video_path")

    # --------------------------------------------------------
    # IMAGE MODE
    # --------------------------------------------------------

    if mode == "image":

        if image_url:

            relative_path = (
                image_url
                .lstrip("/")
                .replace(
                    "/",
                    os.sep
                )
            )

            path = os.path.join(
                BASE_DIR,
                relative_path
            )

            frame = cv2.imread(path)

            if frame is not None:

                ok, buffer = cv2.imencode(
                    ".jpg",
                    frame
                )

                if ok:

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        +
                        buffer.tobytes()
                        +
                        b"\r\n"
                    )

        return

    # --------------------------------------------------------
    # VIDEO MODE
    # --------------------------------------------------------

    if mode == "video":

        if (
            not video_path
            or not os.path.exists(video_path)
        ):
            return

        cap = cv2.VideoCapture(
            video_path
        )

        while True:

            success, frame = cap.read()

            if not success:
                break

            annotated = process_frame(
                frame,
                source_name,
                use_tracking=True,
                confidence=0.25
            )

            ok, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    85,
                ]
            )

            if ok:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    +
                    buffer.tobytes()
                    +
                    b"\r\n"
                )

            time.sleep(0.01)

        cap.release()

        return

    # --------------------------------------------------------
    # LOCAL CAMERA MODE
    #
    # This is only useful when running locally.
    # The deployed browser webcam uses
    # /process_webcam_frame instead.
    # --------------------------------------------------------

    cap = get_camera()

    while True:

        success, frame = cap.read()

        if not success:

            time.sleep(0.1)

            continue

        annotated = process_frame(
            frame,
            "Live Camera",
            use_tracking=True,
            confidence=0.25
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80,
            ]
        )

        if ok:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                +
                buffer.tobytes()
                +
                b"\r\n"
            )


# ============================================================
# GET VIOLATION HISTORY
# ============================================================

def get_violations(limit=100):

    ensure_log()

    rows = []

    with open(
        LOG_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return list(
        reversed(rows)
    )[:limit]


# ============================================================
# AVAILABLE SOURCE FILES
# ============================================================

def available_sources():

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    files = []

    if os.path.isdir(SOURCE_DIR):

        for name in sorted(
            os.listdir(SOURCE_DIR)
        ):

            path = os.path.join(
                SOURCE_DIR,
                name
            )

            if (
                os.path.isfile(path)
                and
                os.path.splitext(
                    name
                )[1].lower()
                in allowed
            ):

                files.append(name)

    return files


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        violations=get_violations(),
        sources=available_sources(),
    )


# ============================================================
# STATISTICS API
# ============================================================

@app.route("/stats")
def stats():

    with state_lock:

        data = dict(state)

    data["recent_violations"] = (
        get_violations(20)
    )

    return jsonify(data)


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/"
            "x-mixed-replace;"
            " boundary=frame"
        ),
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )


# ============================================================
# SET CAMERA MODE
# ============================================================

@app.route(
    "/set_camera",
    methods=["POST"]
)
def set_camera():

    with state_lock:

        state["mode"] = "camera"

        state["source_name"] = (
            "Live Camera"
        )

        state["image_url"] = None

        state["video_path"] = None

    return jsonify(
        {
            "ok": True,
            "mode": "camera",
        }
    )


# ============================================================
# USE BUILT-IN SOURCE FILE
# ============================================================

@app.route(
    "/use_source",
    methods=["POST"]
)
def use_source():

    filename = secure_filename(
        request.form.get(
            "filename",
            ""
        )
    )

    path = os.path.join(
        SOURCE_DIR,
        filename
    )

    if (
        not filename
        or not os.path.isfile(path)
    ):

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Source file not found."
                ),
            }
        ), 400

    ext = os.path.splitext(
        filename
    )[1].lower()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if ext in {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
    }:

        frame = cv2.imread(path)

        if frame is None:

            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "Could not read image."
                    ),
                }
            ), 400

        annotated = process_frame(
            frame,
            filename,
            use_tracking=False,
            confidence=0.25
        )

        out_name = (
            f"processed_"
            f"{int(time.time())}_"
            f"{filename.rsplit('.', 1)[0]}"
            f".jpg"
        )

        out_path = os.path.join(
            RESULT_DIR,
            out_name
        )

        cv2.imwrite(
            out_path,
            annotated
        )

        with state_lock:

            state["mode"] = "image"

            state["source_name"] = (
                filename
            )

            state["image_url"] = (
                "/static/results/"
                + out_name
            )

            state["video_path"] = None

        return jsonify(
            {
                "ok": True,
                "mode": "image",
                "image_url": state[
                    "image_url"
                ],
            }
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if ext in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }:

        with state_lock:

            state["mode"] = "video"

            state["source_name"] = (
                filename
            )

            state["video_path"] = path

            state["image_url"] = None

        return jsonify(
            {
                "ok": True,
                "mode": "video",
            }
        )

    return jsonify(
        {
            "ok": False,
            "error": (
                "Unsupported file type."
            ),
        }
    ), 400


# ============================================================
# UPLOAD FILE
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "No file selected."
                ),
            }
        ), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "No file selected."
                ),
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    ext = os.path.splitext(
        filename
    )[1].lower()

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    if ext not in allowed:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Use JPG/PNG/JFIF or "
                    "MP4/AVI/MOV/MKV/WEBM."
                ),
            }
        ), 400

    unique_name = (
        f"{int(time.time())}_"
        f"{filename}"
    )

    path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    file.save(path)

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    if ext in {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
    }:

        frame = cv2.imread(path)

        if frame is None:

            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "Uploaded image "
                        "could not be read."
                    ),
                }
            ), 400

        annotated = process_frame(
            frame,
            filename,
            use_tracking=False,
            confidence=0.25
        )

        out_name = (
            f"processed_"
            f"{unique_name.rsplit('.', 1)[0]}"
            f".jpg"
        )

        out_path = os.path.join(
            RESULT_DIR,
            out_name
        )

        cv2.imwrite(
            out_path,
            annotated
        )

        with state_lock:

            state["mode"] = "image"

            state["source_name"] = (
                filename
            )

            state["image_url"] = (
                "/static/results/"
                + out_name
            )

            state["video_path"] = None

        return jsonify(
            {
                "ok": True,
                "mode": "image",
                "image_url": state[
                    "image_url"
                ],
            }
        )

    # --------------------------------------------------------
    # VIDEO UPLOAD
    # --------------------------------------------------------

    with state_lock:

        state["mode"] = "video"

        state["source_name"] = (
            filename
        )

        state["video_path"] = path

        state["image_url"] = None

    return jsonify(
        {
            "ok": True,
            "mode": "video",
        }
    )


# ============================================================
# BROWSER WEBCAM FRAME PROCESSING
# ============================================================

@app.route(
    "/process_webcam_frame",
    methods=["POST"]
)
def process_webcam_frame():

    if "frame" not in request.files:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "No webcam frame received."
                ),
            }
        ), 400

    file = request.files["frame"]

    data = file.read()

    if not data:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Empty webcam frame."
                ),
            }
        ), 400

    # Convert uploaded JPEG bytes into OpenCV image
    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify(
            {
                "ok": False,
                "error": (
                    "Could not decode "
                    "webcam frame."
                ),
            }
        ), 400

    try:

        # Browser webcam uses a lower confidence
        # threshold because webcam frames can be
        # compressed or less clear.
        annotated = process_frame(
            frame,
            "Browser Webcam",
            use_tracking=True,
            confidence=0.15
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                75,
            ]
        )

        if not ok:

            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "Could not encode "
                        "detection result."
                    ),
                }
            ), 500

        return Response(
            buffer.tobytes(),
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )

    except Exception as exc:

        return jsonify(
            {
                "ok": False,
                "error": str(exc),
            }
        ), 500


# ============================================================
# RESET VIOLATION LOG
# ============================================================

@app.route(
    "/reset_log",
    methods=["POST"]
)
def reset_log():

    ensure_log()

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        csv.writer(f).writerow(
            [
                "Date",
                "Time",
                "Worker ID",
                "Violation",
                "Source",
            ]
        )

    last_logged.clear()

    return jsonify(
        {
            "ok": True
        }
    )


# ============================================================
# DOWNLOAD CSV
# ============================================================

@app.route("/download_log")
def download_log():

    ensure_log()

    return send_from_directory(
        BASE_DIR,
        "violations.csv",
        as_attachment=True
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False,
        threaded=True,
        use_reloader=False
    )