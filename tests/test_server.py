import unittest
import os
import shutil
import json
from shared.exceptions import *
from server import app, Configs


def post_train_add_json(image_path: str):
    return json.dumps({"image_path": image_path})


class TestRecognitionMethods(unittest.TestCase):
    TEST_IMAGE = "tests/test_data/test_2.jpg"
    TRAINING_DATA = "tests/test_data/recognition/training-data"
    TRAINING_QUEUE = "tests/test_data/recognition/train-images"

    def setUp(self):
        app.config[Configs.TRAINING_DATA] = self.TRAINING_DATA
        app.config[Configs.TRAINING_QUEUE] = self.TRAINING_QUEUE
        self.client = app.test_client()


    def tearDown(self):
        files = os.listdir(self.TRAINING_DATA)
        for f in files:
            shutil.rmtree('%s/%s' % (self.TRAINING_DATA, f))

        files = os.listdir(self.TRAINING_QUEUE)
        for f in files:
            os.remove('%s/%s' % (self.TRAINING_QUEUE, f))


    def test_add_training_image(self):
        response = self.client.post(
          '/train/add', 
          data=post_train_add_json(self.TEST_IMAGE),
          content_type='application/json')
        print(response.data)
        self.assertEqual(response.status_code, 200)
        # expect 2 faces to have been found and added to train queue
        images = os.listdir(self.TRAINING_QUEUE)
        self.assertEqual(len(images), 2)
