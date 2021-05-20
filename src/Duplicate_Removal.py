#!/usr/bin/env python

"""
This script was largely inspired by Adrian Rosebrock from Pyimagesearch who developed this code to look for duplicate images within a dataset. It is an important script to run whenever the dataset has been developed by oneself for two important reasons: 
1. Duplicates within a dataset can lead to implicit bias within the model (and lend to the problem of overfitting) 
2. Many duplicated images can make it difficult for the model to generalise to new data

This is especially important to consider when working with a "smaller" dataset such as the one I have collected here. 

The code below works by transforming images into a numerical representation using some of the simple tools from openCV. It first transforms the data into a greyscale image, resizes it, and runs a popular "hash" algorithm on it to generate its numerical representation. When images are the same, they will have the same numerical representation, and so can be easily found by looping over the images and looking for the hash codes which are the same. This is all done below using looping functions. 
"""

"""
============
Dependencies 
============
"""

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

"""
==============
dhash Function 
==============
"""

"""
This function is applied to every image and essentially converts the image into a numerical representation. 
With this logic, if two images have the same numerical representation, they'll be considered duplicates & one will be removed. 
"""
def dhash(image, hashSize=8):
    # convert the image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #resize the grayscale image (according to the hashSize defined above)
    #Here, we add an additional column to the width to work within the algorithm 
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    
    # compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Apply the hash algorithm defined below which converts the image to a hash
    # Then returns the hash for processing 
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

"""
==========================
Command line functionality
==========================
"""

# Initialize ArgumentParser class
ap = argparse.ArgumentParser()

# Argument 1: Path to the dataset (default is the banksy art) 
ap.add_argument("-d", "--dataset", 
                required=False,
                default = "../data/Scraped_data/Banksy_Streetart", 
                help="path to input dataset")

#Argument 2: Deciding whether to remove the images or just show them (number above 0 deletes duplicates) 
ap.add_argument("-r", "--remove", 
                type=int, default=1,
                help="whether or not duplicates should be removed (i.e., dry run)")

# Parse arguments
args = vars(ap.parse_args())


"""
=============
Main Function
=============

"""
def main():
    
    """
    -----
    Setup
    -----
    """
    #Let the user know the script is starting up 
    print("[INFO] computing image hashes...")

    
    # Initialise the arguments:
    # Connect path to dataset and create empty dictionary to store hashes 
    imagePaths = list(paths.list_images(args["dataset"])) 
    hashes = {}

    print(f"There are {len(imagePaths)} images in this dataset")
    
    
    """
    -----------------------
    Identify the duplicates
    -----------------------
    """

    # Then create a loop to run over the images in the directory
    for imagePath in imagePaths:
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)

    # Here, we create a variable called p, which represents the set of duplicate images 
        p = hashes.get(h, [])     #grab all images with the computed hash (h) 
        p.append(imagePath)       #add the current imagepath to the image  
        hashes[h] = p             #store this in the hashes dictionary 
        
        
    """
    ------------------------------------------------
    Handle the duplicates (either delete or display)
    ------------------------------------------------
    """

    # loop over the image hashes
    for (h, hashedPaths) in hashes.items():
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:
            print (f"It seems we've found {len(hashedPaths)} duplicates of this image")
            # check to see if the user wants to remove the image or not (as defined in commandline) 
            if args["remove"] <= 0:
                # initialize a montage to store all images with the same hash
                montage = None
                # loop over all image paths with the same hash
                for p in hashedPaths:
                    # load the input image and resize it to a fixed width
                    # and heightG
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))
                    # if our montage is None, initialize it
                    if montage is None:
                        montage = image
                    # otherwise, horizontally stack the images
                    else:
                        montage = np.hstack([montage, image])
                # show the montage for the hash
                print("[INFO] hash: {}".format(h))
                cv2.imshow("Montage", montage)
                cv2.waitKey(0)
            
                # If the items are to be removed, the following code will run 
            else:
                # Run through all the image paths with the same hash 
                # Delete all except the first one (so that we will have a copy of the image) 
                for p in hashedPaths[1:]:
                    os.remove(p)
                    
    """
    --------------------
    Finish up the script
    --------------------
    """
    file_count = os.listdir(args["dataset"]) # dir is your directory path
    number_files = len(file_count)
    
    
    print("All the duplicates have been identified and deleted")
    print (f"There are now {number_files} images in this dataset") 
                    
# Close the main function 
if __name__=="__main__":
    main() 


