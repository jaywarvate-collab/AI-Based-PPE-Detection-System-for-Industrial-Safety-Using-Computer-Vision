from ultralytics import YOLO
from pathlib import Path

model = YOLO("models/best.pt")

source = Path("source_files")
results = model.predict(
    source=str(source),
    save=True,
    conf=0.25,
    iou=0.50,
    verbose=True,
)

print("PPE detection completed successfully.")
print("Open the 'runs/detect' folder to see the annotated outputs.")
