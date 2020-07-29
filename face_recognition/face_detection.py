import cv2

class FaceDetectorInterface:
    """
    Class used for identifying faces in images

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
          List of all face coordinates in the original image
        """
        pass
    
    def detect_faces_square(self, filename: str) -> list:
        """
        Returns a list of all square coordinates of each face found

        Parameters
        ----------
        filename : string
          Path to the image file

        Returns
        -------
        list((startX, startY, endX, endY))
          List of all face coordinates in the original image
        """
        pass

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
        pass
        """
        pass

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
        pass
    
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
        pass

    