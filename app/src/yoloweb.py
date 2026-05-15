import numpy as np
from ultralytics import YOLO
import cv2
import math

# VIDEO CAPTURE
cap = cv2.VideoCapture("videos/new.mp4")

# LOAD YOLO MODEL
model = YOLO("../yolo-weight/yolov8l.pt")

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
    #imgRegion = cv2.bitwise_and(img, mask)

    # YOLO DETECTION
    results = model(img, stream=True)

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
                if currentClass == "car":
                    color = (0, 255, 0)

                elif currentClass == "truck":
                    color = (0, 0, 255)

                elif currentClass == "bus":
                    color = (255, 0, 0)

                else:
                    color = (0, 255, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 8)

                # TEXT
                # RECTANGLE
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # TEXT BACKGROUND
                cv2.rectangle(img, (x1, y1 - 60), (x1 + 250, y1), (255, 0, 255), -1)

                # TEXT
                cv2.putText(
                    img,
                    f'{currentClass} {conf}',
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (255, 255, 255),
                    5
                )

    # SHOW OUTPUT
    img = cv2.resize(img, (900, 600))
    cv2.imshow("Image", img)

    cv2.imshow("Masked Region", img)

    # EXIT KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# RELEASE
cap.release()

cv2.destroyAllWindows()
