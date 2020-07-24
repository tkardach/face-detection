from face_detection_mtcnn import MTCNNFaceDetection
from utility import convertToRGB
import matplotlib.pyplot as plt
import cv2
import os
import re


if __name__ == "__main__":
    dirPath = 'train-images'
    images = os.listdir(dirPath)
    detector = MTCNNFaceDetection()

    labels = []
    with open('subjects.txt') as f:
      labels = f.read().splitlines()

    for image in images:
        image_path = dirPath + '/' + image

        faces = detector.extract_faces(image_path)
        
        for face in faces:
            print("Who is this? Give an integer label")
            for (i, item) in enumerate(labels):
              print(i, item)
            plt.imshow(convertToRGB(face[0]))
            plt.show()

            label = input('Enter the label: ')

            dest = 'training-data/s' + str(label)
            if not os.path.exists(dest):
                os.makedirs(dest)

            samples = os.listdir(dest)
            samples = [int(re.search(r'\d+', i).group()) for i in samples]
            
            add_image = dest + '/' + str(max(samples) + 1)+'.jpg'
            print(add_image)
            cv2.imwrite(add_image, face[0])
          
        os.remove(image_path)

        print("Data prepared")
