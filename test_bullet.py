from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load a model

IMAGE = '/home/anastasispk/Dev/ultralytics/Bullet-Holes-6/test/images/target120_jpg.rf.e74cae5da63baf563b166161e51ffda1.jpg'
# IMAGE = '/home/anastasispk/Dev/ultralytics/Bullet-Holes-6/test/images/target56_jpg.rf.7cc1d9ca431c6243708c036301dc4b8f.jpg'
# IMAGE = '/home/anastasispk/Dev/ultralytics/screenshot5.png'
# IMAGE = '/home/anastasispk/Dev/ultralytics/test.png'
# IMAGE = '/home/anastasispk/Dev/ultralytics/bullet_hole_templates.png'

model = YOLO('yolov8l-seg.pt')  # load an official model
model = YOLO('/home/anastasispk/Dev/ultralytics/runs/segment/train/weights/best.pt')  # load a custom model

# Predict with the model
img = cv2.imread(IMAGE)
results = model(IMAGE)  # predict on an image
res_plotted = results[0].plot(img=img, masks=True, boxes=False)
cv2.imshow('', res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()