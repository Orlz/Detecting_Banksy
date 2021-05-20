#!/usr/bin/env python

"""
This script was the initial attempt to download images using the popular google_images_downloader package which can be found at the following address: https://pypi.org/project/google_images_download/

Sadly, this package no longer works after Google restriction updates in 2020. This problem is outlined in the following forum:
https://github.com/hardikvasa/google-images-download/issues/301

"""

# Call the google images downloader 
from google_images_download import google_images_download 
  
# creating object
response = google_images_download.googleimagesdownload() 

#Determine the search terms to be used 
search_queries = [     
'banksy streetart',
'banksy political art'
]

#Function to download the images 
def downloadimages(query):
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit":99,
                 "print_urls":True,
                 "size": "medium",
                 #"aspect_ratio":"square",
                 "output_directory": "../data/Banksy/"}
    try:
        response.download(arguments)
      
    # Handling File NotFound Error    
    except FileNotFoundError: 
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit":4,
                     "print_urls":True, 
                     "size": "medium"}
                       
        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments) 
        except:
            pass

# Driver Code
for query in search_queries:
    downloadimages(query) 
    print() 
    
    
    
#                 "usage_rights": "labeled-for-nocommercial-reuse", 
# 
                 #"save_source": "../data/bansky_sources.txt"
