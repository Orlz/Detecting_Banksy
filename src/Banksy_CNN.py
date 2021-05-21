#!/usr/bin/env python

"""
=============================
Identifying Banksy Street Art
=============================

This script builds an AlexNet inspired Convolutional Neural Network to make a binary classification. It seeks to classify images of renowned Street Artist Banksy against images of other streetart. 

The script has been written to focus on the building of the AlexNet architecture and calls many of the functions used from the Banksy_CNN_utils.py script, found in the utils folder. Many of these functions have been copied or inspired from the functions used for Assignment03 - Impressionist Painters. 

Optional parameters:
    -i  --image_path   <str>    Path to the directory containing all images
    -t  --test_size    <float>  Decimal between 0 and 1 indicating the size of the test set 
    -n  --n_epochs     <int>    The number of epochs to run the model on (recommended: 10) 
    -b  --batch_size   <int>    The batch size to run the model on 

Usage: 

$ python3 src/Banksy_CNN.py

""" 


"""
=======================
Import the Dependencies
=======================

"""
# Operating system
import os
import sys
sys.path.append(os.path.join(".."))

# Data handling tools
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
from contextlib import redirect_stdout

#Functions from the utils folder
import Detecting_Banksy.utils.Banksy_CNN_utils as functions

#Commandline functionality 
import argparse

# Sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# TensorFlow tools
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

#Clear all previous sessions if any exist 
tf.keras.backend.clear_session()


"""
==================
Argparse Arguments
==================

"""
# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Path to the image directory
ap.add_argument("-i", "--image_path",
                type = str,
                required = False,
                help = "Path to the full image directory",
                default = "data/Images")

# Argument 2: Number of epochs
ap.add_argument("-t", "--test_size",
                type = float,
                required = False,
                help = "decimal between 0 and 1 indicating what the test split should be",
                default = 0.2)
    
    
# Argument 3: Number of epochs
ap.add_argument("-n", "--n_epochs",
                type = int,
                required = False,
                help = "The number of epochs to train the model on",
                default = 20)
    
# Argument 4: Batch size
ap.add_argument("-b", "--batch_size",
                type = int,
                required = False,
                help = "The size of the batch on which to train the model",
                default = 32)
    
# Parse arguments
args = vars(ap.parse_args()) 


"""
=============
Main Function
=============

"""
def main():
    
    """
    Create variables with the input parameters
    """
    
    image_path = args["image_path"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    test_size = args["test_size"]
    
        
    """
    Create the out directory, if it doesn't already exist 
    """
    dirName = os.path.join("out")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        
        # print that it has been created
        print("Directory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("Directory " , dirName ,  " already exists")
        
        
    """
    ==============
    Preprocessing
    ==============
    """
    
    print("\n Hello friend. Let's see how unique banksy really is! ")
    print("\n I'm about to initialize the construction of your Convolutional Neural Network model...")     
    print("\n We'll start with the pre-processing. This is likely to take a few minutes.") 
    
    """
    Labelling the data
    """
    print("\n [INFO] Grabbing the labels..") 
    
     # Create the list of label names
    label_names = functions.listdir_nohidden(image_path)
    
    """
    Grabbing the data
    """
    print("\n [INFO] Grabbing the images..")
    #Call the get_data function 
    data, label = functions.get_data(image_path, label_names)
    
    
    """
    Creating the train and test split 
    """
    print("\n [INFO] Splitting the data into its train and test sets") 

    # split data
    (trainX, testX, trainY, testY) = train_test_split(data, 
                                                  label, 
                                                  test_size= test_size)
    
        
    """
    Normalizing and Binarizing 
    """
    print("\n [INFO] Normalizing and Binarizing the data") 
    #Call the normalize_binarize function 
    trainX, trainY, testX, testY = functions.normalize_binarize(trainX, trainY, testX, testY)
    
    
    """
    ===============================
    Building and training the model
    ===============================
    """
    
    #Ensure old session is cleared 
    K.clear_session()

    #Build the model 
    model = Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(Conv2D(32,(11,11),activation='relu',input_shape=(227,227,3)))
    model.add(MaxPooling2D(3,3))

    # Convolutional layer and maxpool layer 2
    model.add(Conv2D(64,(5,5),activation='relu', padding="same"))
    model.add(MaxPooling2D(3,3))

    # Convolutional layer 3 (No maxpool layer) 
    model.add(Conv2D(128,(3,3),activation='relu', padding="same"))

    # Convolutional layer 4(No maxpool layer) 
    model.add(Conv2D(256,(3,3),activation='relu', padding="same"))
    
    # Convolutional layer 5 and maxpool layer 3
    model.add(Conv2D(512,(3,3),activation='relu', padding="same"))
    model.add(MaxPooling2D(3,3))

    # This layer flattens the resulting image array to 1D array
    model.add(Flatten())

    # Hidden layer with 4096 neurons and Rectified Linear Unit activation function
    # We're replicating the AlexNet hence the excessive number of nodes
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    
    # Hidden layer 2 with 4096 neurons and Rectified Linear Unit activation function 
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))

    # Output layer with single neuron which gives 0 for Banksy or 1 for Other Streetrt  
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    
    # Model summary
    model_summary = model.summary()
    
    # name for saving model summary
    model_path = os.path.join("..", "Detecting_Banksy", "out", "model_summary.txt")
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    
    
    # name for saving plot
    plot_path = os.path.join("..", "Detecting_Banksy", "out", "AlexNet_model.png")
    # Visualization of model
    plot_LeNet_model = plot_model(model,
                                  to_file = plot_path,
                                  show_shapes=True,
                                  show_layer_names=True)
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    
    """
    Train the Model
    """
    
    # Train model (call: train_AlexNet_model function) 
    print("\n[INFO] The model's ready so we'll begin training it...\n\n")
    H = functions.train_AlexNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size)
    
    print("\nTraining complete - thanks for your patience! We'll start to evaluate the model's performance") 
    
    
    """
    ====================
    Evaluating the model
    ====================
    """
    
    # Evaluate model
    print("\n[INFO] Below is the classification report. This has been copied into the out directory\n")
    functions.evaluate_model(model, testX, testY, batch_size, label_names)

    # Plot loss/accuracy history of the model
    functions.plot_history(H, n_epochs)

    
    # User message
    print("\n That's you all done - woohoo!\n It looks like our friend Banksy is indeed quite unique!")
    
# Close the main function 
if __name__=="__main__":
    main()  
    
    
    
    
    
