from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weight/yolov8n.pt')
results = model("image/1.png", show=True)
img = results[0].plot()
cv2.imshow("YOLO Result", img)
cv2.waitKey(0)