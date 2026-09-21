import os
import csv
import threading
from datetime import datetime

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify,
    send_file,
)
from ultralytics import YOLO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
SOURCE_DIR = os.path.join(BASE_DIR, "source_files")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

current_source = "Live Camera"
current_camera_index = 0

camera = None
camera_lock = threading.Lock()

inference_lock = threading.Lock()

latest_stats = {
    "workers": 0,
    "safe": 0,
    "violations": 0,
    "compliance": 0.0,
}

violation_csv = os.path.join(BASE_DIR, "violations.csv")

# Prevent the same violation from being recorded on every video frame.
violation_log_lock = threading.Lock()
last_violation_log = {}
VIOLATION_LOG_COOLDOWN = 5  # seconds


# ============================================================
# PPE CLASSES
# ============================================================

VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
}


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():
    if not os.path.exists(violation_csv):
        with open(
            violation_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "Worker ID",
                "Violation",
                "Source"
            ])


initialize_csv()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_name(class_id):
    try:
        return model.names[int(class_id)]
    except Exception:
        return str(class_id)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def update_stats(workers, unsafe_workers):
    safe_workers = max(0, workers - unsafe_workers)

    if workers > 0:
        compliance = (safe_workers / workers) * 100
    else:
        compliance = 0.0

    latest_stats["workers"] = workers
    latest_stats["safe"] = safe_workers
    latest_stats["violations"] = unsafe_workers
    latest_stats["compliance"] = round(compliance, 1)


def log_violation(worker_id, violation, source):
    now = datetime.now()

    with open(
        violation_csv,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            worker_id,
            violation,
            source
        ])


def log_violation_once(worker_id, violation, source):
    """
    Record a violation, but prevent duplicate rows for the same
    worker/violation/source within the cooldown period.
    """
    now = datetime.now()
    key = (
        str(source),
        str(worker_id),
        str(violation)
    )

    with violation_log_lock:
        previous = last_violation_log.get(key)

        if previous is not None:
            elapsed = (now - previous).total_seconds()

            if elapsed < VIOLATION_LOG_COOLDOWN:
                return False

        last_violation_log[key] = now

    try:
        log_violation(
            worker_id,
            violation,
            source
        )

        print(
            f"VIOLATION RECORDED: "
            f"Worker {worker_id} | {violation} | {source}"
        )

        return True

    except Exception as error:
        print(
            "VIOLATION LOG ERROR:",
            repr(error)
        )
        return False


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(
    frame,
    source_name="Live Camera",
    use_tracking=True,
    confidence=0.25,
    imgsz=416
):
    """
    Main PPE detection function.

    Local webcam:
        YOLO tracking + ByteTrack

    Uploaded image / Render webcam:
        YOLO prediction
    """

    global latest_stats

    if frame is None:
        return frame

    # --------------------------------------------------------
    # Resize extremely large frames
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    max_width = 960

    if width > max_width:

        scale = max_width / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    try:

        with inference_lock:

            if use_tracking:

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    imgsz=imgsz,
                    max_det=30,
                    verbose=False
                )

            else:

                results = model.predict(
                    frame,
                    conf=confidence,
                    imgsz=imgsz,
                    max_det=30,
                    verbose=False
                )

    except Exception as error:

        print("YOLO ERROR:", error)

        return frame

    if not results:
        return frame

    result = results[0]

    if result.boxes is None:
        update_stats(0, 0)
        return frame

    boxes = result.boxes

    detections = []

    # --------------------------------------------------------
    # Extract detections
    # --------------------------------------------------------

    for index in range(len(boxes)):

        try:

            xyxy = boxes.xyxy[index].cpu().numpy()

            x1, y1, x2, y2 = map(int, xyxy)

            class_id = int(
                boxes.cls[index].cpu().item()
            )

            confidence_value = float(
                boxes.conf[index].cpu().item()
            )

            class_name = get_class_name(class_id)

            track_id = None

            if (
                use_tracking
                and boxes.id is not None
            ):

                try:

                    track_id = int(
                        boxes.id[index].cpu().item()
                    )

                except Exception:
                    track_id = None

            detections.append({
                "box": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence_value,
                "track_id": track_id
            })

        except Exception:
            continue

    # --------------------------------------------------------
    # Find workers
    # --------------------------------------------------------

    worker_detections = []

    for detection in detections:

        name = detection["class"].lower()

        if name in ["person", "worker"]:

            worker_detections.append(detection)

    workers_count = len(worker_detections)

    # --------------------------------------------------------
    # Associate PPE violations with workers
    # --------------------------------------------------------

    unsafe_workers = 0

    worker_results = []

    for worker_index, worker in enumerate(
        worker_detections,
        start=1
    ):

        worker_box = worker["box"]

        worker_id = worker["track_id"]

        if worker_id is None:
            worker_id = worker_index

        violations_for_worker = []

        for detection in detections:

            class_name = detection["class"]

            if class_name not in VIOLATION_CLASSES:
                continue

            iou = calculate_iou(
                worker_box,
                detection["box"]
            )

            # Also check if the PPE box is inside
            # the worker bounding box.

            wx1, wy1, wx2, wy2 = worker_box

            px1, py1, px2, py2 = detection["box"]

            center_x = (px1 + px2) / 2
            center_y = (py1 + py2) / 2

            inside_worker = (
                wx1 <= center_x <= wx2
                and
                wy1 <= center_y <= wy2
            )

            if iou > 0.02 or inside_worker:

                violations_for_worker.append(
                    class_name
                )

        if violations_for_worker:

            unsafe_workers += 1

            # Record each violation detected for this worker.
            for violation in sorted(set(violations_for_worker)):

                log_violation_once(
                    worker_id,
                    violation,
                    source_name
                )

        worker_results.append({
            "id": worker_id,
            "box": worker_box,
            "violations": violations_for_worker
        })

    # --------------------------------------------------------
    # Update dashboard
    # --------------------------------------------------------

    update_stats(
        workers_count,
        unsafe_workers
    )

    # --------------------------------------------------------
    # Draw worker boxes
    # --------------------------------------------------------

    for worker in worker_results:

        x1, y1, x2, y2 = worker["box"]

        violations = worker["violations"]

        if violations:

            box_color = (0, 0, 255)

            label = (
                f"Worker {worker['id']} | "
                + ", ".join(violations)
            )

        else:

            box_color = (0, 200, 0)

            label = (
                f"Worker {worker['id']} | SAFE"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 30)),
            (
                min(
                    frame.shape[1] - 1,
                    x1 + max(160, len(label) * 8)
                ),
                y1
            ),
            box_color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw PPE violation boxes
    # --------------------------------------------------------

    for detection in detections:

        class_name = detection["class"]

        if class_name not in VIOLATION_CLASSES:
            continue

        x1, y1, x2, y2 = detection["box"]

        confidence_text = (
            f"{class_name} "
            f"{detection['confidence']:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Dashboard overlay
    # --------------------------------------------------------

    overlay_text = (
        f"Workers: {latest_stats['workers']}   "
        f"Safe: {latest_stats['safe']}   "
        f"Violations: {latest_stats['violations']}   "
        f"Compliance: {latest_stats['compliance']}%"
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (min(frame.shape[1] - 10, 650), 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        overlay_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# LOCAL WEBCAM
# ============================================================

def get_camera():

    global camera

    with camera_lock:

        if camera is None:

            camera = cv2.VideoCapture(
                current_camera_index,
                cv2.CAP_DSHOW
            )

            # IMPORTANT:
            # Lower resolution = smoother YOLO processing.

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                480
            )

            camera.set(
                cv2.CAP_PROP_FPS,
                30
            )

            # Reduce internal camera buffer.

            try:
                camera.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )
            except Exception:
                pass

        return camera


def generate_frames():

    global camera

    cap = get_camera()

    frame_count = 0

    last_processed_frame = None

    while True:

        success, frame = cap.read()

        if not success:

            print("Camera frame could not be read.")

            break

        frame_count += 1

        # ----------------------------------------------------
        # PROCESS EVERY 2ND FRAME
        # ----------------------------------------------------

        if frame_count % 2 == 0:

            last_processed_frame = process_frame(
                frame,
                "Live Camera",
                use_tracking=True,
                confidence=0.25,
                imgsz=416
            )

        # ----------------------------------------------------
        # Display latest processed result
        # ----------------------------------------------------

        if last_processed_frame is not None:

            display_frame = last_processed_frame

        else:

            display_frame = frame

        # ----------------------------------------------------
        # JPEG encoding
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            display_frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                75
            ]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# LOCAL VIDEO STREAM
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# STATS
# ============================================================

@app.route("/stats")
def stats():

    return jsonify({
        "workers": latest_stats["workers"],
        "safe": latest_stats["safe"],
        "violations": latest_stats["violations"],
        "compliance": latest_stats["compliance"]
    })


# ============================================================
# SET CAMERA
# ============================================================

@app.route(
    "/set_camera",
    methods=["POST"]
)
def set_camera():

    global current_camera_index
    global camera

    try:

        data = request.get_json(
            silent=True
        ) or {}

        current_camera_index = int(
            data.get("camera", 0)
        )

    except Exception:

        current_camera_index = 0

    with camera_lock:

        if camera is not None:

            camera.release()

            camera = None

    return jsonify({
        "ok": True,
        "camera": current_camera_index
    })


# ============================================================
# BROWSER WEBCAM FOR RENDER
# ============================================================

@app.route(
    "/process_webcam_frame",
    methods=["POST"]
)
def process_webcam_frame():

    if "frame" not in request.files:

        return jsonify({
            "ok": False,
            "error": "No webcam frame received."
        }), 400

    file = request.files["frame"]

    data = file.read()

    if not data:

        return jsonify({
            "ok": False,
            "error": "Empty webcam frame."
        }), 400

    try:

        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "ok": False,
                "error": "Could not decode webcam frame."
            }), 400

        # ----------------------------------------------------
        # Make Render inference lighter
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        max_width = 480

        if width > max_width:

            scale = max_width / width

            frame = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        # ----------------------------------------------------
        # Render uses prediction instead of tracking
        # ----------------------------------------------------

        annotated = process_frame(
            frame,
            "Browser Webcam",
            use_tracking=False,
            confidence=0.20,
            imgsz=320
        )

        success, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                65
            ]
        )

        if not success:

            return jsonify({
                "ok": False,
                "error": "Could not encode detection result."
            }), 500

        return Response(
            buffer.tobytes(),
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache"
            }
        )

    except Exception as error:

        print(
            "WEBCAM INFERENCE ERROR:",
            repr(error)
        )

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


# ============================================================
# IMAGE / VIDEO UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        return jsonify({
            "ok": False,
            "error": "No file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "ok": False,
            "error": "No filename."
        }), 400

    filename = os.path.basename(
        file.filename
    )

    extension = os.path.splitext(
        filename
    )[1].lower()

    input_path = os.path.join(
        SOURCE_DIR,
        filename
    )

    file.save(input_path)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    ]

    if extension in image_extensions:

        image = cv2.imread(
            input_path
        )

        if image is None:

            return jsonify({
                "ok": False,
                "error": "Could not read image."
            }), 400

        result = process_frame(
            image,
            filename,
            use_tracking=False,
            confidence=0.25,
            imgsz=640
        )

        output_name = (
            "result_"
            + os.path.splitext(filename)[0]
            + ".jpg"
        )

        output_path = os.path.join(
            RESULT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            result
        )

        return jsonify({
            "ok": True,
            "type": "image",
            "filename": output_name,
            "url": "/results/" + output_name
        })

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    video_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    ]

    if extension in video_extensions:

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            return jsonify({
                "ok": False,
                "error": "Could not open video."
            }), 400

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 20

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        output_name = (
            "result_"
            + os.path.splitext(filename)[0]
            + ".mp4"
        )

        output_path = os.path.join(
            RESULT_DIR,
            output_name
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            result = process_frame(
                frame,
                filename,
                use_tracking=True,
                confidence=0.25,
                imgsz=416
            )

            writer.write(result)

        cap.release()
        writer.release()

        return jsonify({
            "ok": True,
            "type": "video",
            "filename": output_name,
            "url": "/results/" + output_name
        })

    return jsonify({
        "ok": False,
        "error": "Unsupported file type."
    }), 400


# ============================================================
# RESULT FILES
# ============================================================

@app.route("/results/<filename>")
def results_file(filename):

    safe_filename = os.path.basename(
        filename
    )

    path = os.path.join(
        RESULT_DIR,
        safe_filename
    )

    if not os.path.exists(path):

        return jsonify({
            "error": "Result file not found."
        }), 404

    return send_file(path)


# ============================================================
# USE EXISTING SOURCE FILE
# ============================================================

@app.route(
    "/use_source",
    methods=["POST"]
)
def use_source():

    data = request.get_json(
        silent=True
    ) or {}

    filename = data.get(
        "filename",
        ""
    )

    if not filename:

        return jsonify({
            "ok": False,
            "error": "No source filename."
        }), 400

    filename = os.path.basename(
        filename
    )

    path = os.path.join(
        SOURCE_DIR,
        filename
    )

    if not os.path.exists(path):

        return jsonify({
            "ok": False,
            "error": "Source file not found."
        }), 404

    extension = os.path.splitext(
        filename
    )[1].lower()

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    ]

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in image_extensions:

        image = cv2.imread(path)

        if image is None:

            return jsonify({
                "ok": False,
                "error": "Could not read image."
            }), 400

        result = process_frame(
            image,
            filename,
            use_tracking=False,
            confidence=0.25,
            imgsz=640
        )

        output_name = (
            "source_result_"
            + os.path.splitext(filename)[0]
            + ".jpg"
        )

        output_path = os.path.join(
            RESULT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            result
        )

        return jsonify({
            "ok": True,
            "type": "image",
            "filename": output_name,
            "url": "/results/" + output_name
        })

    return jsonify({
        "ok": False,
        "error": "This source type is not supported here."
    }), 400


# ============================================================
# VIOLATION HISTORY DATA
# ============================================================

@app.route("/violation_history")
def violation_history():

    initialize_csv()

    records = []

    try:

        with open(
            violation_csv,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if any(
                    str(value or "").strip()
                    for value in row.values()
                ):

                    records.append({
                        "Date": row.get("Date", ""),
                        "Time": row.get("Time", ""),
                        "Worker ID": row.get("Worker ID", ""),
                        "Violation": row.get("Violation", ""),
                        "Source": row.get("Source", "")
                    })

    except Exception as error:

        print(
            "HISTORY READ ERROR:",
            repr(error)
        )

        return jsonify({
            "ok": False,
            "records": [],
            "error": str(error)
        }), 500

    # Newest violation first.
    records.reverse()

    response = jsonify({
        "ok": True,
        "records": records
    })

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ============================================================
# DOWNLOAD CSV
# ============================================================

@app.route("/download_log")
def download_log():

    if not os.path.exists(violation_csv):

        initialize_csv()

    response = send_file(
        violation_csv,
        as_attachment=True,
        download_name="violations.csv",
        max_age=0
    )

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ============================================================
# RESET CSV
# ============================================================

@app.route(
    "/reset_log",
    methods=["POST"]
)
def reset_log():

    initialize_csv()

    with open(
        violation_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Worker ID",
            "Violation",
            "Source"
        ])

    return jsonify({
        "ok": True
    })


# ============================================================
# SOURCE FILE LIST
# ============================================================

@app.route("/source_files")
def source_files():

    files = []

    for filename in os.listdir(
        SOURCE_DIR
    ):

        path = os.path.join(
            SOURCE_DIR,
            filename
        )

        if os.path.isfile(path):

            files.append(filename)

    return jsonify(files)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 60)
    print("AI-BASED PPE DETECTION SYSTEM")
    print("=" * 60)
    print(f"Running on port: {port}")
    print("Local webcam mode: OpenCV + YOLO Tracking")
    print("Render webcam mode: Browser + YOLO Prediction")
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )