import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

# VIDEO CAPTURE
cap = cv2.VideoCapture("videos/cars.mp4")

# LOAD YOLO MODEL
model = YOLO("../yolo-weight/yolov8n.pt")

# LOAD MASK
mask = cv2.imread("mask.png")

# CLASS NAMES
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

while True:

    success, img = cap.read()

    # CHECK VIDEO FRAME
    if not success:
        break

    # APPLY MASK
    imgRegion = cv2.bitwise_and(img, mask)

    # YOLO DETECTION
    results = model(imgRegion, stream=True)

    for r in results:

        boxes = r.boxes

        for box in boxes:

            # BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # CONFIDENCE
            conf = math.ceil((box.conf[0] * 100)) / 100

            # CLASS NAME
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            # FILTER VEHICLES
            if (
                currentClass == "car"
                or currentClass == "bus"
                or currentClass == "truck"
                or currentClass == "motorbike"
            ) and conf > 0.3:

                # TEXT
                cvzone.putTextRect(
                    img,
                    f'{currentClass} {conf}',
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    offset=5
                )

                # RECTANGLE
                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=9,
                    rt=2,
                    colorR=(255, 0, 255)
                )

    # SHOW OUTPUT
    cv2.imshow("Image", img)

    cv2.imshow("Masked Region", imgRegion)

    # EXIT KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE
cap.release()

cv2.destroyAllWindows()
