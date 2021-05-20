#!/usr/bin/env python

"""
================================================
Creating a train and test split for your dataset
================================================

This script takes our scraped data folders and creates appropriate training and validation folders for each of the classes. It can be generalised by the user to create train and test splits on other datasets by ammending the root_dir and classes_directory to fit their folder architechture (using the argparse commands).

The script has the following functions:  
1. Create parent folders for training ('train') and validation ('val') data 
2. Create sub-folders within these two folders for each of the defined classes 
3. Shuffle the file names in each path and then apply the designated split (validation_size)
4. Print the number of images for each class in each folder to the terminal 
5. Copy the images from the scraped data folders into their new designated folder

NOTE: This is a one time use script. Any attempts to run it multiple times without deleting the folders which were created in the original run will throw error messages due to the directory already existing. To fix this problem, you can remove the train and val folders from your data directory and re-run the script from new. 

Usage: 
$ python3 Creating_Train_Test_Split.py

"""

"""
------------
Dependencies
------------
"""
# For working with the files 
import os 
import numpy as np
import shutil

# For shuffling the filepaths
import random


"""
==================
Argparse Arguments
==================

"""
# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Size of the validation set 
ap.add_argument("-v", "--validation_size",
                type = float,
                required = False,
                help = "Decimal between 0 and 1 indicating the size of validation set (recommended between 0.15 - 0.3)",
                default = 0.2)
    
# Argument 2: Root directory 
ap.add_argument("-r", "--root_dir",
                type = str,
                required = False,
                help = "Path to the root of where the data is stored",
                default = "data")

# Argument 3: Classes directory 
ap.add_argument("-c", "--classes_dir",
                type = str,
                required = False,
                help = "A list of class folders within the root directory. Should be defined with '/' before each name",
                default = '/Banksy_Streetart', '/Other_Streetart')

# Parse arguments
args = vars(ap.parse_args()) 

"""
=============
Main Function
=============

"""
def main():
    """
    Setting up the script for the desired collection of folders 
    """
    # Connecting the root and class folders 
    root_dir = args["root_dir"]
    classes_dir = [args["classes_dir]
    
    #Defining the validation ratio                     
    val_ratio = args["validation_size"] 
       
                        
    """
    Creating a loop to run through folders and make directories for each class 
    """
    for cls in classes_dir:
        os.makedirs(root_dir +'/train' + cls)
        os.makedirs(root_dir +'/val' + cls)


        # Link to folder to copy images from 
        src = root_dir + cls 
    
        #Get all the filenames in this directory                     
        allFileNames = os.listdir(src)
        #Randomly shuffle them to reduce bias 
        np.random.shuffle(allFileNames)
        #Then partition the splits according to the val_ratio                    
        train_FileNames, val_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames)* (1 - val_ratio))])

        
        """
        Apply the splits to the files   
        """                 
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]

                        
        """
        Print the number of images in each class folder to the terminal  
        """                     
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
                        
        
                        
        """
        Copy the images into the new directory using shutil  
        """  
        for name in train_FileNames:
            shutil.copy(name, root_dir +'/train' + cls)

        for name in val_FileNames:
            shutil.copy(name, root_dir +'/val' + cls)
                        
# Close the main function 
if __name__=="__main__":
    main()  

