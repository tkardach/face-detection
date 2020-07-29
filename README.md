# Facial Detection and Recognition

This repository contains facial detection modules and facial recognition modules I've created through researching the topic. 

The actual facial detection and recognition is not done by me, I have just organized all of my findings into clean classes which are ready to use and can help you with your understanding through your research into the subject.

# Facial Detection

There are 3 methods used for facial detection here.

1. OpenCV's Haar-cascade method 
1. OpenCV's Deep Neural Network method
1. MTCNN (Multi-Task Cascaded Convolutional Neural Network) method


I have personally found that the MTCNN is the most accurate for still images (which is my purpose for the project). As for live detection, you are probably better off using OpenCV's DNN method

# Facial Recognition

The Facial Recognition module uses OpenCV's LBPHFaceRecognizer for training and testing image data. Accumulating training and testing data will be up to you. 

# Add Training Data

I have created a train-images folder, where you can store images and manually add each face to your training data. When we store training images, we conform all faces to a standard size so that we have a common basis for training (and this allows us to recognize faces from different cameras better). I do not claim to have found the best standard for storing these images, but this is something I will be looking into.

To add training data:
1. Add all the images you want to train to the 'train-images' folder (NOTE: these images will be deleted from this folder after training. Make sure to back them up)
1. Run the train_images.py script, it will run through each found face and have you classify the face by the person's name
1. Each trained face will be added to the 'training-data/s#' folder, where '#' is the index of the person in the subject list

# Testing Data

To test your facial recognition, run the face_recognition.py script. This will use all the images from the 'training-data' folder to train the LBPHFaceRecognizer and it will then itterate through the images in the 'testing-data' folder. For each image, it will show the image with all faces encapsulated, and if the recognizer is within a specific confidence level, the name of the person will be shown
