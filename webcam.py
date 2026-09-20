from ultralytics import YOLO
import cv2

# Load trained PPE model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Run YOLO detection
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow(
        "Construction Site PPE Detection",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()