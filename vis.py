import cv2
import numpy as np
from random import randint
import random
import os
from pathlib import Path
random.seed(20)

image_list = os.listdir('Bullet-Holes-6/test/labels')
annotation_file = random.choice(image_list)
image_folder = '/home/anastasispk/Bullet-Holes-6/test/images/'

with open('/home/anastasispk/Dev/ultralytics/Bullet-Holes-6/test/labels/' + annotation_file, 'r') as f:
    labels = f.read().splitlines()
img = cv2.imread(image_folder + Path(image_folder + annotation_file).stem + '.jpg')
h,w = img.shape[:2]

for label in labels:
    class_id, *poly = label.split(' ')
    
    poly = np.asarray(poly,dtype=np.float16).reshape(-1,2) # Read poly, reshape
    poly *= [w,h] # Unscale
    
    cv2.polylines(img, [poly.astype('int')], True, (255,0,0), 2) # Draw Poly Lines
    # cv2.fillPoly(img, [poly.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area


    cv2.imshow('img with poly', img)
    cv2.waitKey(0)