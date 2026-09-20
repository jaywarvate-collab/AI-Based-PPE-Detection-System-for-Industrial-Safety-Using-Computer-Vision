# PPE Detection Dashboard - Run Guide

## 1. Activate the virtual environment

Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
venv\Scripts\activate
```

## 2. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Start the dashboard

```cmd
python app.py
```

Open:
http://127.0.0.1:5000

## 4. Detection modes

The dashboard supports:

- Live webcam
- Uploaded images
- Uploaded videos
- Images/videos already in `source_files`

## 5. Dashboard statistics

- Workers Detected = current Person detections
- PPE Safe = workers without an associated NO-Hardhat, NO-Mask or NO-Safety Vest detection
- Violations = unsafe workers in the current frame
- Compliance = PPE Safe / Workers Detected * 100

The violation logger stores:
Date, Time, Worker ID, Violation and Source.

## Important accuracy note

The supplied `models/best.pt` is the original repository's trained YOLOv8 model. The application improves the application logic by associating PPE detections with the nearest worker instead of treating any NO-* detection anywhere in the frame as a violation for every worker.

No computer-vision system can guarantee 100% accuracy. Lighting, camera angle, occlusion, distance and the training data affect detection quality. For a final-year demonstration, test the model on several representative images/videos and report the observed performance.
