import sys
sys.path.append('../')

import numpy as np
import dlib
import cv2


class FaceAlignment:
    FIVE_LANDMARKS = "face_recognition/model/shape_predictor_5_face_landmarks.dat"
    FIVE_RIGHT_EYE = (0, 2)
    FIVE_LEFT_EYE = (2, 4)

    SIXTY_EIGHT_LANDMARKS = "face_recognition/model/shape_predictor_5_face_landmarks.dat"

    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = dlib.shape_predictor(self.FIVE_LANDMARKS)
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight if desiredFaceHeight is not None else desiredFaceWidth

    def __call__(self, image, shapes):
        shapes = np.array([(point.x, point.y) for point in shapes.parts()])

        leftEye = shapes[self.FIVE_RIGHT_EYE[0]:self.FIVE_RIGHT_EYE[1]]
        rightEye = shapes[self.FIVE_LEFT_EYE[0]:self.FIVE_LEFT_EYE[1]]

        leftEyeCenter = leftEye.mean(axis=0).astype('int')
        rightEyeCenter = rightEye.mean(axis=0).astype('int')        
        

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output

if __name__ == "__main__":
    from face_recognition.face_detection_hog import HOGFaceDetection
    
    #detector = HOGFaceDetection()
    aligner = FaceAlignment()

    detector = dlib.get_frontal_face_detector()
    image = cv2.imread("face_recognition/detection-tests/1_1_.jpg") 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
    # Detect faces with 1 upsample itteration
    faces = self.detector(gray, 1) 

    image = cv2.imread("face_recognition/detection-tests/1_1_.jpg")
    # loop over the face detections
    for rect in faces:
      faceRect = (rect.left(), rect.top(), rect.width(), rect.height())
      faceOrig = image[faceRect[1]:faceRect[1] + faceRect[3], faceRect[0]:faceRect[0]+faceRect[2]]
      faceAligned = aligner(image, rect)
      # display the output images
      cv2.imshow("Original", faceOrig)
      cv2.imshow("Aligned", faceAligned)
      cv2.waitKey(0)


