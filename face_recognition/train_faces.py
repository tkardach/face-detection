import sys
sys.path.append('../')

from face_recognition import FaceRecognizer
from shared.utility import convertToRGB
import matplotlib.pyplot as plt
import cv2
import os
import re


if __name__ == "__main__":
    dirPath = 'train-images'
    images = os.listdir(dirPath)
    
    recognizer = FaceRecognizer()


    for image in images:
        image_path = dirPath + '/' + image

        faces = recognizer.extract_faces(image_path)
        
        for face in faces:
            print("Who is this? Give an integer label")
            for (i, item) in recognizer.get_subjects():
              print(i, item)
            plt.imshow(convertToRGB(face[0]))
            plt.show()

            name = input('Name of person: ')

            index = recognizer.add_subject(name)
              
            if not recognizer.add_training_image(face[0], name):
                print('Failed to add training image')
          
        os.remove(image_path)
