import sys
sys.path.append('../')

from face_detection import FaceDetectorInterface
from mtcnn.mtcnn import MTCNN
from shared.utility import *
from PIL import Image
from shared.image_mod import draw_rectangle
import numpy as np
import cv2


class MTCNNFaceDetection(FaceDetectorInterface):

    """
    Face Detection class using a Multi-Task Cascaded Convolutional Neural Network
    for detection (using mtcnn package)

    ...

    Methods
    -------
    detect_faces(filename)
      Detect all faces in an image and return their locations
    extract_faces(filename)
      Detect and return all faces in an image
    find_faces_on_image(filename)
      Find all faces in an image and return the image with the faces located

    """
    LONG_SIDE_PIXELS = 750

    def __init__(
            self,
            weights_file: str = None,
            min_face_size: int = 20,
            steps_threshold: list = None,
            scale_factor: float = 0.709):
        self.__detector = MTCNN(
            weights_file=weights_file,
            min_face_size=min_face_size,
            steps_threshold=steps_threshold,
            scale_factor=scale_factor)

    def __prepare_image_PIL(self, filename: str):
        original_image = Image.open(filename)

        width, height = original_image.size

        if width > self.LONG_SIDE_PIXELS or height > self.LONG_SIDE_PIXELS:
            return resize_image_PIL(image=original_image, long_side=self.LONG_SIDE_PIXELS), original_image
        else:
            return original_image, original_image

    def __revert_coordinates_PIL(self, coords: np.ndarray, from_image, to_image) -> np.ndarray:
        from_x, from_y = from_image.size
        to_x, to_y = to_image.size

        ratio = to_x / from_x

        return (coords * ratio).astype(int)

    def __prepare_image(self, filename: str):
        width, height = get_image_dimensions(filename=filename)

        if width is None or height is None:
            return None

        original_image = cv2.imread(filename)

        if width > self.LONG_SIDE_PIXELS or height > self.LONG_SIDE_PIXELS:
            return resize_image(image=original_image, long_side=self.LONG_SIDE_PIXELS), original_image
        else:
            return original_image, original_image

    def __revert_coordinates(self, coords: np.ndarray, from_image, to_image) -> np.ndarray:
        from_x, from_y = from_image.shape[:2]
        to_x, to_y = to_image.shape[:2]

        ratio = to_x / from_x

        return (coords * ratio).astype(int)

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
        # load image from file
        resized, original = self.__prepare_image_PIL(filename)
        # detect faces in the image
        results = self.__detector.detect_faces(np.array(resized))
        faces = [i['box'] for i in results]
        faces = np.array(faces)

        w, h = resized.size
        # Convert to common return value (startX,startY,endX,endY)
        # from (startX,startY,width, height)
        faces[:, 2] = faces[:, 0] + faces[:, 2]
        faces[:, 3] = faces[:, 1] + faces[:, 3]
        # For any face rectangle that exceeds the min limit (0), set to 0
        faces[faces < 0] = 0
        # For any face rectangle that exceeds max X limit, set to image width
        faces[:, 0][faces[:, 0] > w] = w
        faces[:, 2][faces[:, 2] > w] = w
        # For any face rectangle that exceeds max Y limit, set to image height
        faces[:, 1][faces[:, 1] > h] = h
        faces[:, 3][faces[:, 3] > h] = h

        # convert the face coordinates back to the original coordinate system
        return self.__revert_coordinates_PIL(faces, resized, original)

    def detect_faces_square(self, filename: str) -> np.ndarray:
        """
        Returns a list of all square coordinates of each face found

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        list((startX, startY, endX, endY))
          List of all face coordinates in the image
        """
        # load image from file
        resized, original = self.__prepare_image_PIL(filename)
        # detect faces in the image
        results = self.__detector.detect_faces(np.array(resized))
        faces = [i['box'] for i in results]
        faces = np.array(faces)

        if len(faces) == 0:
            return faces

        # convert the face coordinates back to the original coordinate system
        faces = self.__revert_coordinates_PIL(faces, resized, original)

        w, h = original.size

        # Set width and height to the larger of the 2 values (make square)
        faces[:,2][faces[:,2] < faces[:,3]] = faces[:,3][faces[:,2] < faces[:,3]]
        faces[:,3][faces[:,3] < faces[:,2]] = faces[:,2][faces[:,3] < faces[:,2]]

        # Convert to common return value (startX,startY,endX,endY)
        # from (startX,startY,width, height)
        faces[:, 2] = faces[:, 0] + faces[:, 2]
        faces[:, 3] = faces[:, 1] + faces[:, 3]
        # For any face rectangle that exceeds the min limit (0), set to 0
        faces[faces < 0] = 0
        # For any face rectangle that exceeds max X limit, set to image width
        faces[:, 0][faces[:, 0] > w] = w
        faces[:, 2][faces[:, 2] > w] = w
        # For any face rectangle that exceeds max Y limit, set to image height
        faces[:, 1][faces[:, 1] > h] = h
        faces[:, 3][faces[:, 3] > h] = h
        
        return faces

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
        list(tuple(np.array, (startX, startY, endX, endY)))
          List of all faces in the form of a tuple with the face image 
          (numpy array) and face coordinates in the original image
        """
        # find coordinates of all faces in image
        faces = self.detect_faces(filename)

        result_list = []

        image = cv2.imread(filename)
        x_pixels, y_pixels = get_image_dimensions(image=image)
        # Extract faces using the faces coordinates
        for face in faces:
            x1, y1, x2, y2 = face
            result_list.append((image[y1:y2, x1:x2], (x1, y1, x2, y2)))
        return result_list

    def extract_faces_square(self, filename: str) -> list:
        """
        Returns a list of tuples with the face image and its 
        square coordinates in the original image.

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        list(tuple(np.array, (startX, startY, endX, endY)))
          List of all faces in the form of a tuple with the face image 
          (numpy array) and face coordinates in the original image
        pass
        """
        # find coordinates of all faces in image
        faces = self.detect_faces_square(filename)
        result_list = []

        image = cv2.imread(filename)
        x_pixels, y_pixels = get_image_dimensions(image=image)
        # Extract faces using the faces coordinates
        for face in faces:
            x1, y1, x2, y2 = face
            result_list.append((image[y1:y2, x1:x2], (x1, y1, x2, y2)))
        return result_list

    def find_faces_on_image(self, filename: str):
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

        img_copy = cv2.imread(filename)

        for face in faces:
            draw_rectangle(img_copy, face)

        return img_copy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    from argparse import RawTextHelpFormatter
    import os
    import timeit

    parser = argparse.ArgumentParser(
        description="""Perform MTCNN Face Detection on image test set

        Test set should be in the following format:

        root/
        |
        |--- detection-tests/
             |--- {test #}_{# of expected faces}_.jpg
             |--- ...
        """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-w', '--weights_file', default=None, type=str, metavar="",
        help='Weight file to use when creating the MTCNN model')
    parser.add_argument(
        '-m', '--min_face_size', default=20, type=int, metavar="",
        help='Minimum size of face to detect')
    parser.add_argument(
        '-st', '--steps_threshold', default=None, type=list, metavar="",
        help='Steps threshold values to use when creating the MTCNN model')
    parser.add_argument(
        '-sf', '--scale_factor', default=0.709, type=float, metavar="",
        help='Scale factor value to scale the image after each itteration')
    parser.add_argument(
        '-s', '--show_faces', default=False, type=bool, metavar="",
        help='If true, it will display the faces found for each test itteration')
    parser.add_argument(
        '-t', '--time_analysis', default=False, type=bool, metavar="",
        help='If true, it will show the time spent detecting faces in an image')
    args = parser.parse_args()

    def test(
        weights_file: str = None,
        min_face_size: int = 20,
        steps_threshold: list = None,
        scale_factor: float = 0.709,
        show_faces: bool=False,
        time_analysis: bool=False):
        dirPath = 'detection-tests'
        images = os.listdir(dirPath)

        total = 0
        failed = 0

        # create the detector, using default weights
        detector = MTCNNFaceDetection(
            weights_file=weights_file,
            min_face_size=min_face_size,
            steps_threshold=steps_threshold,
            scale_factor=scale_factor
        )

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
                plt.imshow(convertToRGB(
                    detector.find_faces_on_image(imagePath)))
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
        weights_file=args.weights_file,
        min_face_size=args.min_face_size,
        steps_threshold=args.steps_threshold,
        scale_factor=args.scale_factor,
        show_faces=args.show_faces,
        time_analysis=args.time_analysis)
