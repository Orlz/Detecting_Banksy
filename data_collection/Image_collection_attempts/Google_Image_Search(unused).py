#!/usr/bin/env python

"""
This script was made functional but sadly not used due to Google restrictions using their API. Google seems to have updated their download API options recently which made the more popular python downloader package (google_images_download) no longer functional with python. This more updated option was attempted but deemed too difficult to get around the API's, therefore a more re-producible solution was found and used in the script named (Google_Image_Downloading.py) 
"""

from google_images_search import GoogleImagesSearch

# you can provide API key and CX using arguments,
# or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX

gis = GoogleImagesSearch('AIzaSyDES8JixhapHuwFWjdkMP5kubVBZhZup2c', 'fe09c1cc1393d3ee8', validate_images=False)

# define search params:
_search_params = {
    'q': "banksy streetart",
    'num': 10,
    'safe': 'off',
    'fileType': 'jpg',
    'imgSize': 'MEDIUM'
    #'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived'
}

# get first 200 images:
gis.search(search_params=_search_params, )

#Then look at the next page 
for image in gis.results():
    image.download('../data/Banksy/streetart')
    image.resize(500, 500)
    


