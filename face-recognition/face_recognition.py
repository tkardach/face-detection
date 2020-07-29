import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from shared.image_mod import *
from shared.utility import convertToRGB
from face_detection_mtcnn import MTCNNFaceDetection
from face_detection_dnn import DNNFaceDetection
from face_detection_haar import HAARFaceDetection
from enum import Enum


class Detector(Enum):
    MTCNN = 0,
    HAAR = 1,
    DNN = 2


class FaceRecognizer:
    """
    """

    FACE_IMAGE_SIZE = 300
    TRAINING_DATA_FOLDER = "training-data"


    def __init__(
        self,
        subjects_file: str='subjects.txt',
        detector_type: Detector=Detector.MTCNN):
        """
        """
        self.subjects_file = subjects_file
        self.__load_subjects()
        if detector_type == Detector.DNN:
            self.detector = DNNFaceDetection()
        elif detector_type == Detector.HAAR:
            self.detector = HAARFaceDetection()
        else:
            self.detector = MTCNNFaceDetection()

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Train the recognizer with our training data
        self.retrain_recognizer()


    def __load_subjects(self):
        """
        Reload the subjects from the subjects file into a list
        """
        with open(self.subjects_file) as f:
            self.subjects = f.read().splitlines()


    def __prepare_face_image_training(self, face_image: np.ndarray) -> np.ndarray:
        """
        Prepare a face image to the recognizer norm (used for training)

        Parameters
        ----------
        face_image: np.array
            The face image we will convert
        
        Returns
        -------
        resized: np.array
            The new image conforming to the FaceRecognizers standard
        
        None
            If the face image is not square, we will not use it
        """
        height, width = face_image.shape[:2]

        if width != height:
            print('prepare_face_image_training: Height %d is not equal to width %d' % (height, width))
            return None
        
        if width < self.FACE_IMAGE_SIZE:
            print('prepare_face_image_training: Image dimension is smaller than minimum %d' % (self.FACE_IMAGE_SIZE))
            return None

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.FACE_IMAGE_SIZE, self.FACE_IMAGE_SIZE))
        

    def __prepare_face_image_testing(self, face_image: np.ndarray) -> np.ndarray:
        """
        Prepare a face image to the recognizer norm (used for recognizing)

        Parameters
        ----------
        face_image: np.array
            The face image we will convert
        
        Returns
        -------
        resized: np.array
            The new image conforming to the FaceRecognizers standard
        """
        height, width = face_image.shape[:2]

        if height == 0 or width == 0:
            None

        print(width, height)
        width_ratio = float(self.FACE_IMAGE_SIZE) / width
        height_ratio = float(self.FACE_IMAGE_SIZE) / height

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=width_ratio, fy=height_ratio)
        

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
            if not dir_name.startswith('s'):
                continue
            
            label = int(dir_name.replace('s', ''))
            subject_dir_path = self.TRAINING_DATA_FOLDER + "/" + dir_name

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

        return faces, np.array(labels)


    def __generate_subject_name(self, name):
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


    def __generate_subject_title(self, name):
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


    def extract_faces(self, image_file):
        return self.detector.extract_faces_square(image_file)


    def get_subjects(self):
        """
        Get a list of all subjects and their index label

        Returns
        -------
        list(tuple) : list(index, name)
            A list of each subject's index and their title
        """
        return [(i, self.__generate_subject_title(name)) for (i, name) in enumerate(self.subjects)]


    def get_subject_index(self, name):
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
        subject_name = self.__generate_subject_name(name)
        if subject_name in self.subjects:
            return self.subjects.index(subject_name)
        
        return -1


    def add_subject(self, name):
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
        # Check if the subject already exists
        subject_name = self.__generate_subject_name(name)
        if subject_name in self.subjects:
            return self.subjects.index(subject_name)
        
        # Make sure our image directory is in-sync with our subjects list
        self.__load_subjects()
        samples = os.listdir(self.TRAINING_DATA_FOLDER)
        if len(samples) != len(self.subjects):
            raise Exception('Number of training directories does not match number of subjects')

        # Append the new subject name and return their index
        with open(self.subjects_file, 'a') as f:
            f.write('\n%s' % subject_name)
        
        os.mkdir('%s/s%d' % (self.TRAINING_DATA_FOLDER, len(self.subjects)))

        self.__load_subjects()
        return self.subjects.index(subject_name)


    def add_training_image(self, face, label_index):
        """
        Add an image to the trainer under the given label. Converts the image to the recognizer's standard

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

        False
            If we could not prepare the face for training:
                1. Face is not square
                2. Face is smaller than specified dimensions
            If the subject does not exist in the training data
        """
        prepared_face = self.__prepare_face_image_training(face)
        
        if prepared_face is None:
            return False

        # Create subject directory if it does not exist
        dest = '%s/s%d' % (self.TRAINING_DATA_FOLDER, label_index)
        if not os.path.exists(dest):
            print('add_training_image: Failed because directory "%s" does not exist' % dest)
            return False

        # Find the largest integer value of the training data
        samples = os.listdir(dest)
        
        image_path = '%s/%d.jpg' % (dest, len(samples))
        cv2.imwrite(image_path, prepared_face)

        return True


    def retrain_recognizer(self):
        """
        Retrain the recognizer using the training data
        """
        faces, labels = self.__prepare_training_data()
        if len(faces) > 0:
            self.recognizer.train(faces, labels)


    def predict(self, image_file, distance: float=50):
        """
        Predict the faces in the image, return the faces 
        """
        faces = self.detector.extract_faces_square(image_file)
        faces = [(self.__prepare_face_image_testing(face[0]), face[1]) for face in faces]

        results = []
        for i in range(len(faces)):
            face = faces[i]
            if face[0] is None:
                continue

            label = self.recognizer.predict(face[0])
            
            print(label)
            if (label[0] == 0): 
                continue
            if (label[1] < distance):
                rect = face[1]
                label_text = self.__generate_subject_title(self.subjects[label[0]])
                results.append((label_text, rect))
            else:
                rect = face[1]
                label_text = self.__generate_subject_title(self.subjects[label[0]])
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
