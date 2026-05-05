import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_video(input_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 🔥 Use AVI first (more stable)
    out = cv2.VideoWriter(
        "output.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()

        out.write(frame)

    cap.release()
    out.release()

    return "output.avi"