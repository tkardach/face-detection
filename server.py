from face_recognition.face_recognition import FaceRecognizer
from shared.utility import is_image
from shared.exceptions import *
from flask import request, jsonify, Response, Flask
from enum import Enum
import os
import json


class Configs(Enum):
    TRAINING_DATA = "TRAINING_DATA"
    TRAINING_QUEUE = "TRAINING_QUEUE"


app = Flask(__name__)


def predict_faces(image_path: str, distance: float):
    # Recognize faces in the image with the given certainty
    results = recognizer.predict(image_path, float(distance))
    names = [name for (name, rect) in results]
    while '' in names:
        names.remove('')

    return names


def add_image_to_training_queue(image_path: str):
    return recognizer.add_image_to_training_queue(image_path)


def get_training_queue():
    faces = os.listdir(recognizer.TRAINING_QUEUE)
    return [os.path.abspath(dest) for dest in faces]


@app.route('/recognition', methods=['POST'])
def recognition():
    if not request.is_json:
        return Response('POST requests must be sent as JSON objects', status=400, mimetype='text/plain')

    request_json = request.get_json()
    image_path = request_json.get('image_path')
    distance = request_json.get('distance')

    try:
        names = predict_faces(image_path, distance)
        return Response(jsonify(names), status=200, mimetype='application/json')
    except FileNotFoundError:
        return Response('%s was not found' % image_path, status=404, mimetype='text/plain')
    except FileNotAnImage:
        return Response('%s is not an image' % image_path, status=400, mimetype='text/plain')
    except:
        return Response('Unknown error occured during predition', status=500, mimetype='text/plain')


@app.route('/train/add', methods=['POST'])
def add_training_image():
    if not request.is_json:
        return Response('POST requests must be sent as JSON objects', status=400, mimetype='text/plain')

    request_json = request.get_json()
    image_path = request_json.get('image_path')

    try:
        train_files = add_image_to_training_queue(image_path)
        return Response(train_files, status=200, mimetype='application/json')
    except FileNotFoundError:
        return Response('%s was not found' % image_path, status=404, mimetype='text/plain')
    except FileNotAnImage:
        return Response('%s is not an image' % image_path, status=400, mimetype='text/plain')
    except Exception as e:
        return Response(str(e), status=500, mimetype='text/plain')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        if not request.is_json:
            return Response('POST requests must be sent as JSON objects', status=400, mimetype='text/plain')

        request_json = request.get_json()
        image_path = request_json.get('image_path')
        subject_name = request_json.get('name')

        try:
            success = recognizer.add_training_image_from_queue(
                image_path, subject_name)
            if success:
                return Response('Image successfully added for subject %s' % subject_name, status=200, mimetype='text/plain')
            else:
                return Response('Failed to add image for subject %s' % subject_name, status=500, mimetype='text/plain')
        except FileNotFoundError:
            return Response('%s was not found' % image_path, status=404, mimetype='text/plain')
        except FileNotAnImage:
            return Response('%s is not an image' % image_path, status=400, mimetype='text/plain')
        except:
            return Response('Unknown error occured during predition', status=500, mimetype='text/plain')
    elif request.method == 'GET':
        faces = get_training_queue()
        return Response(jsonify(faces), status=200, mimetype='application/json')
    else:
        return Response('Only GET and POST methods are supported', status=400, mimetype='text/plain')


training_data = None
training_queue = None

if Configs.TRAINING_DATA in app.config:
    training_data = app.config[Configs.TRAINING_DATA]
if Configs.TRAINING_QUEUE in app.config:
    training_data = app.config[Configs.TRAINING_QUEUE]

# Initialize the recognizer
recognizer = FaceRecognizer(
    training_data=training_data,
    training_queue=training_queue
)

if __name__ == "__main__":
    app.run(debug=True)
