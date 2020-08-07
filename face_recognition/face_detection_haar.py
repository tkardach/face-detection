import sys
sys.path.append('../')

from shared.image_mod import draw_rectangle
from face_recognition.face_detection import FaceDetectorInterface
from shared.utility import get_image_dimensions, resize_image, convertToRGB
import cv2
import numpy as np


MIN_SIZE_RATIO = 25
MAX_SIZE_RATIO = 5
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 5
CASCADE = "haarcascade_frontalface_alt.xml"


class HAARFaceDetection(FaceDetectorInterface):
    """
    Face Detection class using OpenCV's Haarcascade classifier

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
            f_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
            scale_factor: float=1.05,
            min_neighbors: int=4,
            min_size: float=(30,30)):
        """Constructor for Deep Nerual Network face detection method

        Parameters
        ----------
        f_cascade : CascadeClassifier
            Cascade function to use for detection
        colored_img : Image
            The image used to find faces
        scale_factor : float
            The amount to reduce the image by after each itteration
        min_neighbors : int
            Number of detections to be found before declaring a positive find
        min_size : (int, int)
            Minimum detection size (should be larger for larger images)
        """
        self.f_cascade = f_cascade
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def __prepare_image(self, filename: str):
        width, height = get_image_dimensions(filename=filename)

        if width is None or height is None:
            return None

        original_image = cv2.imread(filename)

        if width > self.LONG_SIDE_PIXELS or height > self.LONG_SIDE_PIXELS:
            return resize_image(image=original_image, long_side=self.LONG_SIDE_PIXELS), original_image
        else:
            return original_image, original_image

    def __revert_coordinates(self, coords: list, from_image, to_image):
        from_x, from_y = get_image_dimensions(image=from_image)
        to_x, to_y = get_image_dimensions(image=to_image)

        ratio = to_x / from_x

        return (coords * ratio).astype("int")

    def detect_faces(self, filename: str) -> list:
        """
        Returns a list of all rectangle coordinates of each face found

        Parameters
        ----------

        Returns
        -------
        rectangles
            An array of rectangles of the face locations
        """
        """
        Returns a list of all square coordinates of each face found

        Parameters
        ----------

        Returns
        -------
        rectangles
            An array of squares of the face locations
        """
        # Convert to grayscale (OpenCV only works on grayscale images)
        resized, original = self.__prepare_image(filename)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        # Detect faces
        faces = self.f_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)


        faces = np.array(faces)
        if (faces.size == 0):
            return faces

        faces = self.__revert_coordinates(faces, resized, original)

        h, w = original.shape[:2]

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
        # Extract faces using the faces coordinates
        for face in faces:
            x1, y1, x2, y2 = face
            result_list.append((image[y1:y2, x1:x2], face))
        return result_list


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
        description="""Perform DNN Face Detection on image test set

        Test set should be in the following format:

        root/
        |
        |--- detection-tests/
             |--- {test #}_{# of expected faces}_.jpg
             |--- ...
        """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-sf', '--scale_factor', default=1.05, type=float, metavar="",
        help='The amount to scale the image by for each face detection itteration. value > 1')
    parser.add_argument(
        '-n', '--min_neighbors', default=4, type=int, metavar="",
        help='The minimum amount of neighbors a detection should have before it is accepted.')
    parser.add_argument(
        '-m', '--min_size', default=(20,20), type=tuple, metavar="",
        help='The minimum size of face to detect')
    parser.add_argument(
        '-s', '--show_faces', default=False, type=bool, metavar="",
        help='If true, it will display the faces found for each test itteration')
    parser.add_argument(
        '-ta', '--time_analysis', default=False, type=bool, metavar="",
        help='If true, it will show the time spent detecting faces in an image')
    args = parser.parse_args()

    def test(
            scale_factor: float=1.05,
            min_neighbors: int=4,
            min_size: tuple=(20,20),
            show_faces: bool=False,
            time_analysis: bool=False):
        dirPath = 'detection-tests'
        images = os.listdir(dirPath)

        total = 0
        failed = 0

        # create the detector, using default weights
        detector = HAARFaceDetection(
            f_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE),
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size)

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
            scale_factor= args.scale_factor,
            min_neighbors= args.min_neighbors,
            min_size= args.min_size,
            show_faces= args.show_faces,
            time_analysis= args.time_analysis)
