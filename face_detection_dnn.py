from face_detection import FaceDetectorInterface
import cv2
import numpy as np
from utility import *
from PIL import Image
from image_mod import draw_rectangle


class DNNFaceDetection(FaceDetectorInterface):
    """
    Face Detection class using OpenCV's Deep Neural Network method

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
            confidence: float = 0.5,
            use_caffe: bool = False,
            use_skin: bool = False,
            skin_perc: float = 0.4):
        """Constructor for Deep Nerual Network face detection method

        Parameters
        ----------
        confidence : float
            Filter detected faces by this confidence value (default 50% confidence)
        use_caffe : bool
            If True, use the Caffe framework for creating the DNN model. If false, use TensorFlow.
        """
        self.confidence = confidence
        self.use_skin = use_skin
        self.skin_perc = skin_perc
        if use_caffe:
            self.model = cv2.dnn.readNetFromCaffe(
                './model/deploy.prototxt', './model/res10_300x300_ssd_iter_140000.caffemodel')
        else:
            self.model = cv2.dnn.readNetFromTensorflow(
                './model/opencv_face_detector_uint8.pb', './model/opencv_face_detector.pbtxt')

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

    def __detect_skin(self, img):
        """Find skin pixels and return the skin mask.

        Parameters
        ----------
        img: ndarray
        Image to detect skin pixels

        Returns
        -------
        ndarray
        Skin mask where non-zero values indicate skin
        """
        if img.size == 0:
            return np.array([])

        min_range = np.array([0,133,77], np.uint8)
        max_range = np.array([235,173,127], np.uint8)

        conv_image = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        skin = cv2.inRange(conv_image,min_range,max_range)

        return skin

    def __get_skin_for_face_rows(self, row, image):
        x1, y1, x2, y2 = row
        skin = self.__detect_skin(image[y1:y2, x1:x2])
        if skin.size == 0:
            return 0
        return skin[skin > 0].size / skin.size

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
        resized, original = self.__prepare_image(filename)
        h, w = resized.shape[:2]

        # detect faces in the image
        blob = cv2.dnn.blobFromImage(
            resized, 1.05, (w, h), (104.0, 177.0, 123.0))

        self.model.setInput(blob)
        detections = self.model.forward()

        # Extract confidence (index 2) and rectangle coordinates (index 3-7)
        results = detections[0, 0, :, 2:7]
        # Filter out results below our confidence threshold
        filtered_results = results[results[:, 0] > self.confidence]
        # Convert rectangle coordinates (1:5) to our coordinate system as int type
        faces = (filtered_results[:, 1:5] *
                 np.array([w, h, w, h])).astype("int")
        # For any face rectangle that exceeds the min limit (0), set to 0
        faces[faces < 0] = 0
        # For any face rectangle that exceeds max X limit, set to image width
        faces[:, 0][faces[:, 0] > w] = w
        faces[:, 2][faces[:, 2] > w] = w
        # For any face rectangle that exceeds max Y limit, set to image height
        faces[:, 1][faces[:, 1] > h] = h
        faces[:, 3][faces[:, 3] > h] = h

        # If we are using skin as parameter, filter by skin pixels
        if faces.size > 0 and self.use_skin:
            skin = np.apply_along_axis(self.__get_skin_for_face_rows, 1, faces, resized)
            faces = faces[skin > self.skin_perc,:]

        # convert the face coordinates back to the original coordinate system
        return self.__revert_coordinates(faces, resized, original)

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
        description="""Perform DNN Face Detection on image test set

        Test set should be in the following format:

        root/
        |
        |--- tests/
             |--- {test #}_{# of expected faces}_.jpg
             |--- ...
        """, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-c', '--conf', default=0.5, type=float, metavar="",
        help='Face detection confidence must be higher than this value to be accepted')
    parser.add_argument(
        '-t', '--tensor', default=True, type=bool, metavar="",
        help='If true, use TensorFlow framework for DNN model. If false, use Caffe')
    parser.add_argument(
        '-us', '--use_skin', default=False, type=bool, metavar="",
        help="""If true, will use the percentage of skin found in a found face to 
        determine whether to accept the face or not""")
    parser.add_argument(
        '-sp', '--skin_perc', default=0.4, type=float, metavar="",
        help="""The found face must contain a percentage of skin pixels greater than 
        this amount to be accepted""")
    parser.add_argument(
        '-s', '--show_faces', default=False, type=bool, metavar="",
        help='If true, it will display the faces found for each test itteration')
    parser.add_argument(
        '-ta', '--time_analysis', default=False, type=bool, metavar="",
        help='If true, it will show the time spent detecting faces in an image')
    args = parser.parse_args()

    def test(
            confidence: float,
            tensor: bool,
            show_faces: bool,
            use_skin: bool,
            skin_perc: float,
            time_analysis: bool=False):
        dirPath = 'tests'
        images = os.listdir(dirPath)

        total = 0
        failed = 0

        # create the detector, using default weights
        detector = DNNFaceDetection(
            confidence=confidence,
            use_caffe=not tensor,
            use_skin=use_skin,
            skin_perc=skin_perc
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
                plt.imshow(convertToRGB(detector.find_faces_on_image(imagePath)))
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
        confidence=args.conf,
        tensor=args.tensor,
        use_skin=args.use_skin,
        skin_perc=args.skin_perc,
        show_faces=args.show_faces,
        time_analysis=args.time_analysis)
