import unittest
import os
from face_recognition.face_recognition import *
from shared.utility import is_image, does_file_exist
import shutil


class TestRecognitionMethods(unittest.TestCase):
    TRAINING_DATA = "tests/test_data/recognition/training-data"
    TRAINING_QUEUE = "tests/test_data/recognition/train-images"
    TEST_IMAGE = "tests/test_data/test_2.jpg"

    TEST_SUBJECTS = [
      "test_one",
      "test_two",
      "test_three"
    ]

    TEST_SUBJECTS_TITLES = [
      "Test One",
      "Test Two",
      "Test Three"
    ]


    def setUp(self):
        for subject in self.TEST_SUBJECTS:
            try:
                os.mkdir('%s/%s' % (self.TRAINING_DATA, subject)) 
            except FileExistsError:
                pass

        self.recognizer = FaceRecognizer(
          training_data=self.TRAINING_DATA,
          training_queue=self.TRAINING_QUEUE
        )


    def tearDown(self):
        files = os.listdir(self.TRAINING_DATA)
        for f in files:
            shutil.rmtree('%s/%s' % (self.TRAINING_DATA, f))

        files = os.listdir(self.TRAINING_QUEUE)
        for f in files:
            os.remove('%s/%s' % (self.TRAINING_QUEUE, f))
    

    def test_generate_subject_name(self):
        test = "Subject Name"
        converted = FaceRecognizer.generate_subject_name(test)
        self.assertEqual(converted, 'subject_name')


    def test_generate_subject_title(self):
        test = "subject_name"
        converted = FaceRecognizer.generate_subject_title(test)
        self.assertEqual(converted, 'Subject Name')


    def test_get_subjects(self):
        subjects = self.recognizer.get_subjects()
        subject_names = [i[1] for i in subjects]
        self.assertEqual(len(subjects), len(self.TEST_SUBJECTS))
        for title in self.TEST_SUBJECTS_TITLES:
            self.assertIn(title, subject_names)


    def test_get_subject_index(self):
        subjects = self.recognizer.get_subjects()
        for title in self.TEST_SUBJECTS_TITLES:
            index = self.recognizer.get_subject_index(title)
            self.assertIn((index, title), subjects)


    def test_add_subject(self):
        test_subject = "test_new"
        test_title = "Test New"

        new_index = self.recognizer.add_subject(test_title)
        subjects = self.recognizer.get_subjects()

        self.assertIn((new_index, test_title), subjects)


    def test_add_empty_subject(self):
        new_index = self.recognizer.add_subject('')

        self.assertEqual(new_index, -1)


    def test_add_training_image(self):
        face_images = self.recognizer.add_image_to_training_queue(self.TEST_IMAGE)

        for face_path in face_images:
            face = cv2.imread(face_path)
            self.recognizer.add_training_image(face, self.TEST_SUBJECTS[0])

        faces = os.listdir('%s/%s' % (self.TRAINING_DATA, self.TEST_SUBJECTS[0]))
        self.assertEqual(len(faces), 2)


    def test_add_training_image_new_subject(self):
        subject_title = 'Test New'
        subject = 'test_new'
        face_images = self.recognizer.add_image_to_training_queue(self.TEST_IMAGE)

        for face_path in face_images:
            face = cv2.imread(face_path)
            self.recognizer.add_training_image(face, subject_title)

        faces = os.listdir('%s/%s' % (self.TRAINING_DATA, subject))
        self.assertEqual(len(faces), 2)


    def test_add_training_image_new_subject_typeerror(self):
        with self.assertRaises(TypeError):
            self.recognizer.add_training_image('', '')


    def test_add_image_to_queue(self):
        face_images = self.recognizer.add_image_to_training_queue(self.TEST_IMAGE)

        self.assertEqual(len(face_images), 2)
        for image in face_images:
            self.assertTrue(is_image(image))


    def test_add_image_to_queue_throws_on_bad_image(self):
        with self.assertRaises(FileNotAnImage):
            self.recognizer.add_image_to_training_queue('')


    def test_add_training_image_from_queue(self):
        face_images = self.recognizer.add_image_to_training_queue(self.TEST_IMAGE)

        images_created = 0
        self.assertEqual(len(face_images), 2)
        for image in face_images:
            self.assertTrue(is_image(image))
            self.assertTrue(self.recognizer.add_training_image_from_queue(image, self.TEST_SUBJECTS_TITLES[0]))
            self.assertFalse(does_file_exist(image))
            images_created += 1
            faces = os.listdir('%s/%s' % (self.TRAINING_DATA, self.TEST_SUBJECTS[0]))
            self.assertEqual(len(faces), images_created)


    def test_add_training_image_from_queue(self):
        with self.assertRaises(FileNotAnImage):
            self.recognizer.add_training_image_from_queue('', self.TEST_SUBJECTS_TITLES[0])



if __name__ == '__main__':
    unittest.main()