import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_mod import *
from utility import convertToRGB
from face_detection_mtcnn import MTCNNFaceDetection


subjects = []
with open('subjects.txt') as f:
    subjects = f.read().splitlines()


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels= []

    for dir_name in dirs:
        if not dir_name.startswith('s'):
            continue
        
        label = int(dir_name.replace('s', ''))
        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith('.'):
                continue
            image_path = subject_dir_path + "/" + image_name

            face = cv2.imread(image_path)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            if face is not None:
                faces.append(gray)
                labels.append(label)

    return faces, labels
            
def predict(image_file, recognizer, detector):
    print("Predicting faces in image %s" % image_file)
    faces = detector.extract_faces(image_file)

    img = cv2.imread(image_file)
    for i in range(len(faces)):
        face = faces[i]
        label = recognizer.predict(cv2.cvtColor(face[0], cv2.COLOR_BGR2GRAY))
        print(label)
        rect = face[1]
        label_text = subjects[label[0]]
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[3]+5)

    return img

if __name__ == "__main__":
    dirPath = 'tests'
    images = os.listdir(dirPath)
    detector = MTCNNFaceDetection()

    for image in images:
        image_path = dirPath + '/' + image

        print("Preparing data...")
        faces, labels = prepare_training_data("training-data")

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))

        img = predict(image_path, face_recognizer, detector)

        plt.imshow(convertToRGB(img))
        plt.show()

        print("Data prepared")
