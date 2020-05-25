import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread("DSC_1094.JPG")
img_recolored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_mask = np.zeros((4000, 6000, 3), dtype="uint8") + 20
img_modified = cv2.add(img_recolored, img_mask)
img_gray = cv2.cvtColor(img_modified, cv2.COLOR_BGR2GRAY)

classifier = cv2.CascadeClassifier("E:/Programming/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml")
faces = classifier.detectMultiScale(img_gray)
for face in faces:
    x, y, w, h = face
    cv2.rectangle(img_modified, (x, y), (x+w, y+h), (255, 0, 0), 20)


plt.imshow(img_modified, "gray")
plt.show()
