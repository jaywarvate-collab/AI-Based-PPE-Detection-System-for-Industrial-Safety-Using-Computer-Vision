from ultralytics import YOLO

# Load trained PPE detection model
model = YOLO("models/best.pt")

# Input video
video_path = "source_files/hardhat.mp4"

# Run detection
results = model.predict(
    source=video_path,
    save=True,
    conf=0.25
)

print("Video detection completed successfully!")