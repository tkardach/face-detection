import unittest
import numpy as np
from shared.exceptions import *
from face_recognition.face_detection_hog import HOGFaceDetection


class TestRecognitionMethods(unittest.TestCase):
    TEST_IMAGE = "tests/test_data/test_2.jpg"

    def setUp(self):
        self.detector = HOGFaceDetection()


    def tearDown(self):
        pass


    def test_detect_faces(self):
        faces = self.detector.detect_faces(self.TEST_IMAGE)
        self.assertEqual(len(faces), 2)


    def test_detect_faces_throw_if_file_not_exist(self):
        with self.assertRaises(FileNotAnImage):
            faces = self.detector.detect_faces('')

    
    def test_extract_faces(self):
        faces = self.detector.extract_faces(self.TEST_IMAGE)
        self.assertEqual(len(faces), 2)
        for face in faces:
            self.assertTrue(isinstance(face[0], np.ndarray))
            self.assertTrue(isinstance(face[1], tuple))

            
    def test_extract_faces_throw_if_file_not_exist(self):
        with self.assertRaises(FileNotAnImage):
            faces = self.detector.extract_faces('')

    
