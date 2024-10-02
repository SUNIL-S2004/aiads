from ultralytics import YOLO
import numpy
import cv2

model = YOLO("D:/aidas/best.pt")
model.predict(source="0", show = True, conf = 0.7 )
