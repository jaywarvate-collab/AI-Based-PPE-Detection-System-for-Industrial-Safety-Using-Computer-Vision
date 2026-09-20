from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime

# Load model
model = YOLO("models/best.pt")

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# CSV file
log_file = "violations.csv"

# Create CSV if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date",
            "Time",
            "Violation"
        ])

print("Violation logging started.")
print("Press Q to quit.")

# Prevent logging the same violation every frame
last_logged = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        source=frame,
        persist=True,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    detected_classes = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        detected_classes.append(class_name)

    # PPE violations
    violations = []

    if "NO-Hardhat" in detected_classes:
        violations.append("NO-Hardhat")

    if "NO-Mask" in detected_classes:
        violations.append("NO-Mask")

    if "NO-Safety Vest" in detected_classes:
        violations.append("NO-Safety Vest")

    # Log violations
    current_time = datetime.now()

    for violation in violations:

        last_time = last_logged.get(violation)

        # Log same violation only once every 5 seconds
        if (
            last_time is None
            or (current_time - last_time).total_seconds() >= 5
        ):

            with open(log_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    current_time.strftime("%d-%m-%Y"),
                    current_time.strftime("%H:%M:%S"),
                    violation
                ])

            last_logged[violation] = current_time

            print(
                f"Violation logged: {violation}"
            )

    # Display status
    if violations:

        cv2.putText(
            annotated_frame,
            "SAFETY VIOLATION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        y = 80

        for violation in violations:

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
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "PPE Safety Monitor + Violation Logger",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Monitoring stopped.")
print(f"Violation log saved to: {log_file}")