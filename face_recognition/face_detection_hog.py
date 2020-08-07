import sys
sys.path.append('../')

from face_recognition.face_detection import FaceDetectorInterface
from face_recognition.face_alignment import FaceAlignment
from shared.utility import is_image, does_file_exist
from shared.image_mod import draw_rectangle
from shared.exceptions import *
import numpy as np
import cv2
import dlib


class HOGFaceDetection(FaceDetectorInterface):
    """
    """
    LONG_SIDE_PIXELS = 500
    FIVE_LANDMARKS_PREDICTOR = "face_recognition/model/shape_predictor_5_face_landmarks.dat"

    def __init__(self):
        """
        """
        self.detector = dlib.get_frontal_face_detector()
        self.aligner = FaceAlignment()
        self.predictor = dlib.shape_predictor(self.FIVE_LANDMARKS_PREDICTOR)


    def __detect_faces_dlib(self, filename: str) -> list:
        if not does_file_exist(filename):
            raise FileNotFoundError('__detect_faces_dlib: File %s does not exist' % filename)
        if not is_image(filename):
            raise FileNotAnImage('__detect_faces_dlib: File %s is not an image' % filename)
        
        image = cv2.imread(filename) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
          
        # Detect faces with 1 upsample itteration
        rects = self.detector(gray, 1) 

        return (rects, image, gray)


    def detect_faces(self, filename: str) -> list:
        """
        Returns a list of all rectangle coordinates of each face found

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        list((startX, startY, endX, endY))
          List of all face coordinates in the image
        """ 
        (rects, image, gray) = self.__detect_faces_dlib(filename)
        
        # Return face rectangles as numpy array
        return np.array([(rect.left(), rect.top(), rect.width(), rect.height()) for rect in rects])


    def extract_faces(self, filename: str) -> list:
        """
        Returns a list of tuples with the face image and its 
        rectangle coordinates in the original image.

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        list(tuple(np.array, (startX, startY, width, height)))
          List of all faces in the form of a tuple with the face image 
          (numpy array) and face coordinates in the original image
        """
        (rects, image, gray) = self.__detect_faces_dlib(filename)

        ret = []
        for rect in rects:
            shapes = self.predictor(gray, rect)
            aligned_face = self.aligner(image, shapes)

            ret.append((aligned_face, (rect.left(), rect.top(), rect.width(), rect.height())))
        
        return ret


    def show_faces(self, filename: str):
        """
        Returns the original image with all found faces located with a rectangle

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        image : np.array
          The original image with all found faces located with a rectangle
        """
        faces = self.detect_faces(filename)

        image = cv2.imread(filename)

        for face in faces:
            rect = (face[0], face[1], face[0] + face[2], face[1] + face[3])
            draw_rectangle(image, rect)
        
        return image



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    from argparse import RawTextHelpFormatter
    import os
    import timeit

    parser = argparse.ArgumentParser(
        description="""Perform DNN Face Detection on image test set

        Test set should be in the following format:

        root/
        |
        |--- detection-tests/
             |--- {test #}_{# of expected faces}_.jpg
             |--- ...
        """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-s', '--show_faces', default=False, type=bool, metavar="",
        help='If true, it will display the faces found for each test itteration')
    parser.add_argument(
        '-ta', '--time_analysis', default=False, type=bool, metavar="",
        help='If true, it will show the time spent detecting faces in an image')
    args = parser.parse_args()

    def test(
            show_faces: bool,
            time_analysis: bool=False):
        dirPath = 'detection-tests'
        images = os.listdir(dirPath)

        total = 0
        failed = 0

        # create the detector, using default weights
        detector = HOGFaceDetection()

        time = []

        for image in images:
            split = image.split('_')
            test = int(split[0])
            numFaces = int(split[1])

            imagePath = dirPath + '/' + image

            if time_analysis:
                time_num = timeit.timeit(lambda: detector.detect_faces(imagePath), number=5)
                time.append('Time for %s : %f seconds' % (image, time_num / 5))

            # detect faces in the image
            faces = detector.detect_faces(imagePath)

            if show_faces:
                plt.imshow(convertToRGB(detector.show_faces(imagePath)))
                plt.show()

            total += numFaces

            if numFaces > len(faces):
                failed += numFaces - len(faces)
            elif len(faces) > numFaces:
                failed += len(faces) - numFaces

        success = total - failed
        percentage = (float(success) / total) * 100
        print("Total %d\tFailed %d\tSuccess %d\tPercent %f" %
              (total, failed, success, percentage))

        if time_analysis:
            print(*time, sep='\n')
            
    test(
        show_faces=args.show_faces,
        time_analysis=args.time_analysis)
