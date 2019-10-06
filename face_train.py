import os
import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

y_lables = []
x_train = []
current_id = 0
lable_id = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()

            if not label in lable_id:
                lable_id[label] = current_id
                current_id += 1
            id_ = lable_id[label]
            print(lable_id)

            pil_image = Image.open(path).convert("L") #grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.AFFINE)
            image_array = np.array(final_image, "uint8")
            print(image_array)
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_lables.append(id_)

print(y_lables)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickel.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("training.yml")