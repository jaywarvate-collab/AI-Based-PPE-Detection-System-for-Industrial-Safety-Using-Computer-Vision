from ultralytics import YOLO
import cv2

# Load trained PPE model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("PPE Safety Monitor started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Run YOLO
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Get detected class names
    detected_classes = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    # Check PPE violations
    violations = []

    if "NO-Hardhat" in detected_classes:
        violations.append("NO HARDHAT")

    if "NO-Mask" in detected_classes:
        violations.append("NO MASK")

    if "NO-Safety Vest" in detected_classes:
        violations.append("NO SAFETY VEST")

    # Draw YOLO boxes
    annotated_frame = results[0].plot()

    # Display status
    if violations:
        status = "SAFETY VIOLATION"
    else:
        status = "PPE STATUS: OK"

    cv2.putText(
        annotated_frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if violations else (0, 255, 0),
        2
    )

    # Display individual violations
    y = 80

    for violation in violations:
        cv2.putText(
            annotated_frame,
            "WARNING: " + violation,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        y += 30

    # Show window
    cv2.imshow(
        "Construction Site Safety Monitor",
        annotated_frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("PPE Safety Monitor stopped.")