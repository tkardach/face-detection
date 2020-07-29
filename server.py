from face_recognition.face_recognition import FaceRecognizer
from shared.utility import is_image
import flask
from flask import request, jsonify
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
    <p>A prototype API for distant reading of science fiction novels.</p>'''

@app.route('/recognition', methods=['POST'])
def recognition():
    if not request.is_json:
        return 'POST requests must be sent as JSON objects'
    
    request_json = request.get_json()
    image_file = request_json.get('image_path')
    distance = request_json.get('distance')

    if not is_image(image_file):
        return 'image_path must be a valid path to an image'
      
    # Recognize faces in the image with the given certainty
    results = recognizer.predict(image_file, float(distance))
    names = [name for (name, rect) in results]
    while '' in names:
        names.remove('')
    
    return jsonify(names)

@app.route('/train/add', methods=['POST'])
def add_training_image():
    if not request.is_json:
        return 'POST requests must be sent as JSON objects'

    request_json = request.get_json()
    image_file = request_json.get('image_path')

    if not is_image(image_file):
        return 'image_path must be a valid path to an image'

    return jsonify(recognizer.add_image_to_training_queue(image_file))


@app.route('/train', methods=['GET','POST'])
def train():
    if request.method == 'POST':
        if not request.is_json:
            return 'POST requests must be sent as JSON objects'

        request_json = request.get_json()
        image_file = request_json.get('image_path')
        subject_name = request_json.get('name')
        
        success = recognizer.add_training_image_from_queue(image_file, subject_name)
        if success:
            return 'Image successfully added for subject %s' % subject_name
        else:
            return 'Failed to add image for subject %s' % subject_name
    elif request.method == 'GET':
        faces = os.listdir(recognizer.TRAINING_QUEUE)
        faces = [os.path.abspath(dest) for dest in faces]
        return jsonify(faces)
    else:
        image_list = os.listdir()
        return 'Only GET and POST methods are supported'


# Initialize the recognizer
recognizer = FaceRecognizer()

app.run()