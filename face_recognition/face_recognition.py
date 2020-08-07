import sys
sys.path.append('../')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from shared.image_mod import *
from shared.utility import convertToRGB, is_image
from shared.exceptions import *
from shared.exceptions import *
from face_recognition.face_detection_hog import HOGFaceDetection
from enum import Enum
import uuid 


class FaceRecognizer:
    """
    """

    FACE_IMAGE_SIZE = 300
    TRAINING_DATA_FOLDER = "face_recognition/training-data"
    TRAINING_QUEUE = "face_recognition/train-images"


    def __init__(
        self,
        training_data: str = None,
        training_queue: str = None):
        """
        """
        if training_data is not None:
            self.TRAINING_DATA_FOLDER = training_data
        if training_queue is not None:
            self.TRAINING_QUEUE = training_queue
        
        self.detector = HOGFaceDetection()

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Initialize the subjects string
        self.__load_subjects()
        # Train the recognizer with our training data
        self.retrain_recognizer()


    @staticmethod
    def generate_subject_name(name: str):
        """
        Generate a normalized subject name

        Parameters
        ----------
        name: str
            Subject name to be converted to the normal convention

        Returns
        -------
        str
            Subject name in the normalized format
        """
        return str.lower(name.replace(' ', '_'))


    @staticmethod
    def generate_subject_title(name: str):
        """
        Get the subject title from the normalized subject name

        Parameters
        ----------
        name: str
            Normalized subject name to be converted to a title

        Returns
        -------
        str
            Title name of the given subject name string
        """
        return name.replace('_',' ').title()


    def __load_subjects(self):
        """
        Reload the subjects from the subjects file into a list
        """
        self.subjects = os.listdir(self.TRAINING_DATA_FOLDER)


    def __prepare_training_data(self):
        """
        Load all of the training data images and their respective labels

        Returns
        -------
        list(tuple): list(face, label)
            A list containing faces with their respective label
        """
        # Get all subject training folders
        dirs = os.listdir(self.TRAINING_DATA_FOLDER)

        faces = []  # faces will store all grayscale faces used for training
        labels= []  # labels will

        # For each subject, gather their faces and respective labels
        for dir_name in dirs:
            subject_dir_path = self.TRAINING_DATA_FOLDER + "/" + dir_name

            subject_images_names = os.listdir(subject_dir_path)
            
            label = self.get_subject_index(dir_name)
            if label == -1:
                continue

            for image_name in subject_images_names:
                if image_name.startswith('.'):
                    continue
                image_path = subject_dir_path + "/" + image_name

                face = cv2.imread(image_path)
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                if face is not None:
                    faces.append(gray)
                    labels.append(label)

        return faces, np.array(labels)


    def extract_faces(self, image_file: str):
        return self.detector.extract_faces(image_file)


    def get_subjects(self):
        """
        Get a list of all subjects and their index label

        Returns
        -------
        list(tuple) : list(index, name)
            A list of each subject's index and their title
        """
        return [(i, FaceRecognizer.generate_subject_title(name)) for (i, name) in enumerate(self.subjects)]


    def get_subject_index(self, name: str):
        """
        Return the index/label of the subject

        Parameters
        ----------
        name: str
            Name of the subject being found

        Returns
        -------
        index: int
            Index of the given subject

        -1
            If the subject was not found
        """
        # Check if the subject already exists
        subject_name = FaceRecognizer.generate_subject_name(name)
        if subject_name in self.subjects:
            return self.subjects.index(subject_name)
        
        return -1


    def add_subject(self, name: str):
        """
        Add a new subject to the trainer and return their index label

        Parameters
        ----------
        name: str
            Name of the subject to add to the trainer
        
        Returns
        -------
        index: int
            Index/Label of the newly added subject
            Index/Label of the existing subject if already exists

        Raises
        ------
        Exception
            If the number of subject directories in the training folder != number of subjects
        """
        if name == '':
            return -1

        # Make sure our image directory is in-sync with our subjects list
        self.__load_subjects()

        # Check if the subject already exists
        subject_name = FaceRecognizer.generate_subject_name(name)
        if subject_name in self.subjects:
            return self.subjects.index(subject_name)
        
        samples = os.listdir(self.TRAINING_DATA_FOLDER)
        if len(samples) != len(self.subjects):
            raise Exception('Number of training directories does not match number of subjects')

        os.mkdir('%s/%s' % (self.TRAINING_DATA_FOLDER, subject_name))

        self.__load_subjects()
        self.retrain_recognizer()
        return self.subjects.index(subject_name)


    def add_training_image(self, face: np.ndarray, subject_name: str):
        """
        Add an image to the trainer under the given label. 

        Parameters
        ----------
        face: numpy.ndarray
            The face image being added to the trainer
        label_index: int
            The subject label corresponding to the face

        Returns
        -------
        True
            If the image was successfully added to the training data
        """
        if not isinstance(face, np.ndarray):
            raise TypeError('add_training_image: parameter "face" must be of type numpy.ndarray')

        # Make sure our image directory is in-sync with our subjects list
        self.__load_subjects()

        if subject_name not in self.subjects:
            self.add_subject(subject_name)

        # Check if the subject already exists
        label = FaceRecognizer.generate_subject_name(subject_name)

        # Create subject directory if it does not exist
        dest = '%s/%s' % (self.TRAINING_DATA_FOLDER, label)
        if not os.path.exists(dest):
            raise FileNotFoundError('add_training_image: Failed because directory "%s" does not exist' % dest)

        # Find the largest integer value of the training data
        samples = os.listdir(dest)
        
        image_path = '%s/%d.jpg' % (dest, len(samples))
        cv2.imwrite(image_path, face)

        return True


    def add_image_to_training_queue(self, image_file: str):
        if not is_image(image_file):
            raise FileNotAnImage('add_image_to_training_queue: %s either does not exist, or is not an image' % image_file)
        
        image_name = os.path.basename(image_file)
        
        faces = self.extract_faces(image_file)

        files = []
        for face in faces:
            name = uuid.uuid4().hex[:6].upper()
            dest = '%s/%s.jpeg' % (self.TRAINING_QUEUE, name)
            cv2.imwrite(dest, face[0])
            files.append(os.path.abspath(dest))

        return files


    def add_training_image_from_queue(self, face_file: str, subject_name: str):
        if not is_image(face_file):
            raise FileNotAnImage('add_training_image_from_queue: %s is not an image' % face_file)

        face = cv2.imread(face_file)

        added = self.add_training_image(face, subject_name)

        if added:
            os.remove(face_file)
        
        return added


    def retrain_recognizer(self):
        """
        Retrain the recognizer using the training data
        """
        faces, labels = self.__prepare_training_data()
        if len(faces) > 0:
            self.recognizer.train(faces, labels)


    def predict(self, image_file: str, distance: float=45):
        """
        Predict the faces in the image, return the faces 
        """
        faces = self.detector.extract_faces(image_file)

        results = []
        for i in range(len(faces)):
            face = faces[i]
            if face[0] is None:
                continue

            label = self.recognizer.predict(face[0])
            
            if (label[0] == 0): 
                continue
            if (label[1] < distance):
                rect = face[1]
                label_text = FaceRecognizer.generate_subject_title(self.subjects[label[0]])
                results.append((label_text, rect))
            else:
                rect = face[1]
                label_text = FaceRecognizer.generate_subject_title(self.subjects[label[0]])
                results.append(('', rect))

        return results


if __name__ == "__main__":
    dirPath = 'testing-data'
    images = os.listdir(dirPath)
    recognizer = FaceRecognizer()

    for image in images:
        image_path = dirPath + '/' + image

        img = cv2.imread(image_path)
        results = recognizer.predict(image_path)

        for result in results:
            h, w = img.shape[:2]
            rect = result[1]
            draw_rectangle(img, rect)
            draw_text(img, result[0], rect[0], rect[3]+5, 10)

        plt.imshow(convertToRGB(img))
        plt.show()
