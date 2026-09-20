from ultralytics import YOLO
import cv2

# Load PPE model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Worker + PPE monitoring started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Track objects
    results = model.track(
        source=frame,
        persist=True,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # Draw YOLO detections
    annotated_frame = result.plot()

    workers = 0
    violations = 0

    # Store detected classes
    detected = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        detected.append(class_name)

        # Count people
        if class_name == "Person":
            workers += 1

    # Check PPE violations
    violation_names = []

    if "NO-Hardhat" in detected:
        violation_names.append("NO-Hardhat")

    if "NO-Mask" in detected:
        violation_names.append("NO-Mask")

    if "NO-Safety Vest" in detected:
        violation_names.append("NO-Safety Vest")

    violations = len(violation_names)

    # Display worker count
    cv2.putText(
        annotated_frame,
        f"Workers: {workers}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Display status
    if violations > 0:

        cv2.putText(
            annotated_frame,
            "SAFETY VIOLATION",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        y = 110

        for violation in violation_names:

            cv2.putText(
                annotated_frame,
                f"WARNING: {violation}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            y += 30

    else:

        cv2.putText(
            annotated_frame,
            "PPE STATUS: OK",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display tracked worker IDs
    if result.boxes.id is not None:

        ids = result.boxes.id.int().cpu().tolist()

        y = 180

        for track_id in ids:

            cv2.putText(
                annotated_frame,
                f"Tracked ID: {track_id}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y += 25

    # Show camera
    cv2.imshow(
        "Construction Site Worker PPE Monitor",
        annotated_frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Worker PPE monitoring stopped.")