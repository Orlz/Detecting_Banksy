#!/usr/bin/env python

"""
==================
Script Description
==================
This script has been largely inspire by Adrian Rosebrock from pyimagesearch. He developed the loops used to convert urls, while I adapted the script to work through a directory of txt files rather than just one text file and append the images from all the files in a directory into one data folder. The script was also ammended to have command line functionality with a main function and argparse arguments, making it robust enough to work with any image directory of urls. 

What does this script do?
The script takes a directory of txt files containing urls and runs them through a process whereby the script requests the url link, saves the image attached to the url as a jpg file, and writes this into a variable named f. 

A new loop then goes through all the downloaded images files and looks to see whether this image can be used by openCV. It asks the file to open using cv2.imread and if nothing is returned, then this file is deleted from the output folder. 

The output of the script is then a folder full of jpg images ready to be pre-processed

Credit to Adrian Rosebrock(source: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/?fbclid=IwAR3_R3XzW6LlnPtc-nHxx-9UMEXQOvJsQaHEtuSsrsx-WKSp3toZ25O6sug#download-the-code) 

"""

"""
============
Dependencies 
============
"""
# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os

"""
==========================
Command line functionality
==========================
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", 
                required=False,
                default = "../Styling_Banksy/data_collection/URL_txt_files/Banksy_URLs/",
                help="path to directory containing txt files with the image URLs")
ap.add_argument("-o", "--output", 
                required=False,
                default = "../Styling_Banksy/data/Banksy/Banksy_Streetart",
                help="path to output directory of images")
args = vars(ap.parse_args())


"""
=============
Main Function
=============

"""
def main():
    
    """
    ------------------------------
    Extract files in the directory
    ------------------------------
    """
    
    # Connect the directory to the argparse directory 
    directory = args["output"]
    
    #Create an empty list for filenames 
    files = []
 
    #for each file in the directory 
    for filename in os.listdir(directory):
        # connect the filename to the variable f
        f = os.path.join(directory, filename)
        #if this filename is a file 
        if os.path.isfile(f):
            #Then add it to the files list
            files.append(f)
            
    """
    ------------------------------------------------------
    Get the urls from each file and save to a list of rows
    ------------------------------------------------------
    """
    
    rows = []
    for file in files: 
        urls = open(file).read().strip().split("\n")
    
        for url in urls:
            rows.append(url)
        
    total = 0

    """
    -------------------------
    Make the urls into images
    -------------------------
    """
    # loop over each url in the txt file 
    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)
            # save the image as a jpg file 
            p = os.path.sep.join([args["output"], "{}.jpg".format(
                str(total).zfill(8))])
            f = open(p, "wb")
            f.write(r.content)
            f.close()
            
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
            
        # Informs the reader if the files are problematic by saying that it'll skip this url
        # Some errors are expected because it's a messy dataset we're working with 
        except:
            print("[INFO] error downloading {}...skipping".format(p))
        
        
    """
    -------------------------
    Delete problematic images
    -------------------------
    """
    # loop over the image paths we've just downloaded
    for imagePath in paths.list_images(args["output"]):
        # initialize if the image should be deleted or not
        delete = False
        # try to load the image
        try:
            image = cv2.imread(imagePath)
            # if the image is `None` then we could not properly load it from disk, so delete it
            if image is None:
                delete = True
        # if OpenCV cannot load the image then the image is likely corrupt so we should delete it
        except:
            print("Except")
            delete = True
        # check to see if the image should be deleted
        if delete:
            print("[INFO] deleting {}".format(imagePath))
            os.remove(imagePath)

    """
    --------------------
    Finish up the script
    --------------------
    """
    output = args["output"]
    print(f"Your images have been converted and saved in {output}.") 
                    
# Close the main function 
if __name__=="__main__":
    main() 
