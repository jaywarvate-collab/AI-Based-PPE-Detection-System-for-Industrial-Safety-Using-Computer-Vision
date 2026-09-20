from ultralytics import YOLO
import cv2

# Load trained PPE model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Worker tracking started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Run YOLO tracking
    results = model.track(
        source=frame,
        persist=True,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Check whether tracking IDs exist
    if results[0].boxes.id is not None:

        tracking_ids = results[0].boxes.id.int().cpu().tolist()

        # Display number of tracked objects
        worker_count = len(tracking_ids)

        cv2.putText(
            annotated_frame,
            f"Tracked Objects: {worker_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Display IDs
        y = 80

        for track_id in tracking_ids:

            cv2.putText(
                annotated_frame,
                f"Worker ID: {track_id}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y += 30

    else:

        cv2.putText(
            annotated_frame,
            "No workers detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Display camera
    cv2.imshow(
        "Construction Site Worker Tracking",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Worker tracking stopped.")