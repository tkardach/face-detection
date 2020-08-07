from face_recognition.face_recognition import FaceRecognizer
from shared.utility import is_image
from shared.exceptions import *
import flask
from flask import request, jsonify
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def predict_faces(image_path: str, distance: float):
    # Recognize faces in the image with the given certainty
    results = recognizer.predict(image_file, float(distance))
    names = [name for (name, rect) in results]
    while '' in names:
        names.remove('')
    
    return names


def add_image_to_training_queue(image_path: str):
    return recognizer.add_image_to_training_queue(image_file)


def get_training_queue():
    faces = os.listdir(recognizer.TRAINING_QUEUE)
    return [os.path.abspath(dest) for dest in faces]


@app.route('/recognition', methods=['POST'])
def recognition():
    if not request.is_json:
        return 'POST requests must be sent as JSON objects'
    
    request_json = request.get_json()
    image_file = request_json.get('image_path')
    distance = request_json.get('distance')
    
    try:
        names = predict_faces(image_file, distance)
        return jsonify(names)
    except FileNotFoundError:
        return '%s was not found' % image_file
    except FileNotAnImage:
        return '%s is not an image' % image_file
    except:
        return 'Unknown error occured during predition'
    

@app.route('/train/add', methods=['POST'])
def add_training_image():
    if not request.is_json:
        return 'POST requests must be sent as JSON objects'

    request_json = request.get_json()
    image_file = request_json.get('image_path')

    try:
        train_files = add_image_to_training_queue(image_file)
        return jsonify(train_files)
    except FileNotFoundError:
        return '%s was not found' % image_file
    except FileNotAnImage:
        return '%s is not an image' % image_file
    except:
        return 'Unknown error occured during predition'


@app.route('/train', methods=['GET','POST'])
def train():
    if request.method == 'POST':
        if not request.is_json:
            return 'POST requests must be sent as JSON objects'

        request_json = request.get_json()
        image_file = request_json.get('image_path')
        subject_name = request_json.get('name')
        
        try:
            success = recognizer.add_training_image_from_queue(image_file, subject_name)
            if success:
                return 'Image successfully added for subject %s' % subject_name
            else:
                return 'Failed to add image for subject %s' % subject_name
        except FileNotFoundError:
            return '%s was not found' % image_file
        except FileNotAnImage:
            return '%s is not an image' % image_file
        except:
            return 'Unknown error occured during predition'
    elif request.method == 'GET':
        faces = get_training_queue()
        return jsonify(faces)
    else:
        return 'Only GET and POST methods are supported'


# Initialize the recognizer
recognizer = FaceRecognizer()

app.run()