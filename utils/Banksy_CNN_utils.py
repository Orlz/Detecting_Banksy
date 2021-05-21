#!/usr/bin/env python

""" 
============================================================
UTILITIES SCRIPT: Banksy CNN tools for Image Classification 
============================================================

This script contains 6 functions necessary for running the Banksy_CNN.py script. 

Many of these functions have been adapted from Assignment 03, the Impressionist Painters CNN. They demonstrate how common functions used to process data can be used to quickly run complex CNN models where the main script just needs to build the model, define the data directories and call the functions from here. 

There are also 2 additional functions to adapt into the script if the user would prefer to use their own train and test sets 


"""

"""
------------
Dependencies
------------
"""
#connecting to the image directory 
import os
import sys
sys.path.append(os.path.join(".."))
import glob


#image data manipulation
import cv2
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout

#creating plots
import matplotlib.pyplot as plt

#sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense, 
                                     Dropout) # dropout layers added
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD



"""
------------------------------------------------
1. Function for extracting labels from filenames 
------------------------------------------------
"""

def listdir_nohidden(path):
    """
    This function extracts the label names by listing the names of folders within the training directory. 
    It does not list the names of hidden files which begin with a fullstop (.)  
    """
    # Create empty list
    label_names = []
    
    # For every name in training directory
    for name in os.listdir(path):
        # If it does not start with . (which hidden files do)
        if not name.startswith('.'):
            label_names.append(name)
            
    return label_names

"""
------------------------------------------------------
2. Function for grabbing and processing the image data 
------------------------------------------------------
"""

def get_data (image_path, label_names):
    """
    The function grabs data from the "Images" folder, resizes it, transforms it into a numpy array, and stacks it vertically
    data = images saved as NumPy arrays  
    label = classification label  
    """
    # Create empty array and list
    data = np.empty((0, 227, 227, 3))
    label = []
    
    
    # Loop through images in training data
    for name in label_names:
        images = glob.glob(os.path.join(image_path, name, "*.jpg"))
        
        # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image with the specified dimensions using cv2's resize function
            # We'll use the dimensions used by the AlexNet architechture 
            resized_img = cv2.resize(loaded_img, (227, 227), interpolation = cv2.INTER_AREA)
        
            # Create array of image
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the trainX
            data = np.vstack((data, image_array))
            
            # Append the label name to the trainY list
            label.append(name)
        
    return data, label


"""
-----------------------------
3. Normalizing and Binarizing 
-----------------------------
"""

def normalize_binarize(trainX, trainY, testX, testY):
    """
    This function applies normalization to the training and test data (trainX and testX)
    It also applies binarization to the training and test labels so they can be used in the model (trainY and testY)
    """
    
    # Normalize training and test data
    trainX = trainX.astype("float") / 255.
    testX = testX.astype("float") / 255.
    
    # Binarize training and test labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return trainX, trainY, testX, testY



"""
----------------------
4. Training the model 
----------------------
"""
def train_AlexNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size):
    """
    Training the AlexNet model on the training data and validating it on the test data.
    """
    # fit model and save fitting hisotry
    H = model.fit(trainX, trainY,
                    validation_data=(testX, testY), 
                    batch_size=32, 
                    epochs=20, 
                    verbose=1)
    return H

"""
------------------------
5. Evaluating the model 
------------------------
"""
    
def evaluate_model(model, testX, testY, batch_size, label_names):
    """
    This function evaluates the trained model and saves the classification report in the out folder.
    """
    
    #Ensure the label_names are correct 
    label_names = ["Banksy_Streetart", "Other_Streetart"]

    predictions = model.predict(testX, batch_size= batch_size)
    classification = classification_report(testY,
                                          predictions.round(), #This round is important for forcing predictions to be an integer
                                          target_names=label_names)
    
    print(classification)

    
    # name for saving report
    report_path = os.path.join("..","Detecting_Banksy", "out", "classification_report.txt")
    
    # Save classification report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1)))
                                                  
    
    print(f"\n[INFO] Classification report is saved as '{report_path}'.")


"""
------------------------
6. Plotting the results  
------------------------
"""

def plot_history(H, n_epochs):
    """
    This function plots the loss/accuracy of the model during training and saves this as png file in the out folder.
    It uses matplotlib tools to create the plot.
    """
    # name for saving output
    figure_path = os.path.join("..","Detecting_Banksy", "out", "model_history.png")
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")
    
    
"""
Additional Functions to be used if the Creating_Train_Test_Split.py script is used to split the folders 
"""

## To use a dataset which has been split into the the train and val folders, the user should ammend the Banksy_CNN script at line 173 
# That is, they should take out the train_test split function and replace it with the following 2 functions which link to the train and test folders 

"""
--------------------------------------------
1. Function to create the training data sets 
--------------------------------------------
"""

#The user needs to have defined a "train_data" variable which links to the train folder (e.g. data/train) 
def create_trainX_trainY(train_data, label_names):
    """
    The function creates the trainX and trainY sets to be as follows:
    trainX = training data 
    trainY = training labels  
    """
    # Create empty array and list
    trainX = np.empty((0, min_height, min_width, 3))
    trainY = []
    
    # Loop through images in training data
    for name in label_names:
        images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image with the specified dimensions using cv2's resize function
            resized_img = cv2.resize(loaded_img, (227, 227), interpolation = cv2.INTER_AREA)
        
            # Create array of image
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the trainX
            trainX = np.vstack((trainX, image_array))
            
            # Append the label name to the trainY list
            trainY.append(name)
        
    return trainX, trainY


"""
----------------------------------------
2. Function to create the test data sets 
----------------------------------------
"""
#The user needs to have defined a "test_data" variable which links to the test folder (e.g. data/val) 
def create_testX_testY(test_data, label_names):
    """
    The function creates the testX and testY sets to be as follows:
    testX = validation data 
    testY = validation labels  
    """
    # Create empty array and list
    testX = np.empty((0, min_height, min_width, 3))
    testY = []
    
    # Loop through images in test data
    for name in label_names:
        images = glob.glob(os.path.join(test_data, name, "*.jpg"))
    
    # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image
            resized_img = cv2.resize(loaded_img, (227, 227), interpolation = cv2.INTER_AREA)
        
            # Create array
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the testX
            testX = np.vstack((testX, image_array))
            # Append the label name to the testY list
            testY.append(name)
        
    return testX, testY


