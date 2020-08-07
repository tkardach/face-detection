import cv2
import os
import numpy as np
from PIL import Image


def convertToRGB(img: np.ndarray):
    """Returns the image in RGB color format

    Parameters
    ----------
    img : Image
        The image to convert
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convert_within_bounds(coords: list, lower_bound: int, upper_bound: int) -> list:
    """Convert all values in coords to be within the lower and upper bounds

    Parameters
    ----------
    coords : list(number)
      List of values to be converted
    lower_bound : number
      Lower bound for numbers in coords
    upper_bound : number
      Upper bound for numbers in coords

    Returns
    -------
    list(number)
      List of converted values
    """
    return [lower_bound if i < lower_bound
            else upper_bound if i > upper_bound
            else i 
            for i in coords]


def get_image_dimensions(filename: str=None, image=None) -> tuple:
    """Return the width and height of the image

    Parameters
    ----------
    filename : string
      Name of the image file

    Returns
    -------
    width, height : int, int
      Width and Height of the image in pixels

    None, None
      If an IOError occurs while trying to read open the file as an Image
    """
    if image is None and filename is None:
        return None, None
    try:
        if image is None:
            image = cv2.imread(filename)
        h,w = image.shape[:2]
        return w, h
    except IOError:
        return None, None


def resize_image(filename: str=None, image=None, width: int=0, height: int=0, long_side: int=0):    
    """Return a resized image according to the dimensions used

    Parameters
    ----------
    image : numpy.array
      Image to resize
    filename : string
      Name of the image file
    width : int
      Desired width of the resized image. Maintains aspect ratio if height not specified
    height : int
      Desired height of the resized image. Maintains aspect ratio if width not specified
    long_side : int
      Desired length of the resized image's longest size. Width and height are ignored if long_side used.
      Aspect ratio is maintained when using long_side

    Returns
    -------
    ndarray
      Returns the resized image

    None
      If an IOError occurs while trying to read open the file as an Image
    """
    if image is None and filename is None:
        return None
    try:
        if image is None:
            image = cv2.imread(filename)
        
        x_pixels, y_pixels = get_image_dimensions(image=image)


        ratio = 1
        if long_side != 0:
            if x_pixels > y_pixels:
                ratio = long_side / x_pixels
            else:
                ratio = long_side / y_pixels
        elif width == 0 and height != 0:
            ratio = height / y_pixels
        elif height == 0 and width != 0:
            ratio = width / x_pixels
        elif height != 0 and width != 0:
            return cv2.resize(image, (int(width), int(height)))

        return cv2.resize(image, (int(x_pixels * ratio), int(y_pixels * ratio))) 
    except IOError:
        return None


def resize_image_PIL(filename: str=None, image=None, width: int=0, height: int=0, long_side: int=0):    
    """Return a resized image according to the dimensions used

    Parameters
    ----------
    image : numpy.array
      Image to resize
    filename : string
      Name of the image file
    width : int
      Desired width of the resized image. Maintains aspect ratio if height not specified
    height : int
      Desired height of the resized image. Maintains aspect ratio if width not specified
    long_side : int
      Desired length of the resized image's longest size. Width and height are ignored if long_side used.
      Aspect ratio is maintained when using long_side

    Returns
    -------
    Image
      Returns the resized image

    None
      If an IOError occurs while trying to read open the file as an Image
    """
    if image is None and filename is None:
        return None
    try:
        if image is None:
            image = Image.open(filename)
        
        (x_pixels, y_pixels) = image.size

        ratio = 1
        if long_side != 0:
            if x_pixels > y_pixels:
                ratio = long_side / x_pixels
            else:
                ratio = long_side / y_pixels
        elif width == 0 and height != 0:
            ratio = height / y_pixels
        elif height == 0 and width != 0:
            ratio = width / x_pixels
        elif height != 0 and width != 0:
            return image.resize((int(width), int(height)))

        return image.resize((int(x_pixels * ratio), int(y_pixels * ratio))) 
    except IOError:
        return None


def does_file_exist(file_path: str):
    exists = os.path.isfile(file_path)
    if not exists:
        return False
    return True


def is_image(file_path: str):
    try:
        if not does_file_exist(file_path):
            return False
        image = Image.open(file_path)
        image.close()
    except IOError:
        return False

    return True