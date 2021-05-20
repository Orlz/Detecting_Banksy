[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com)

# Banksy Street Art 

<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/graffiti_icon.png"/></div>

## A Binary CNN Classifier to detect Banksy's Streetart among other Street Graffiti

This assignment aims to pull the techniques gathered throughout the visual analytics course together into one line of exploration. The project takes the topic of graffiti artist Banksy and considers whether machine learning can shed new light into our understanding of his work and the hype surrounding his anonymous façade. Namely it asks, how unique is Banksy’s street art and how easily could it be faked by another? 

A key focus of the assignment is demonstrating how one could go about gathering an image dataset to be used in computer vision tasks. This is in recognition that many computer vision projects use the same large datasets such as MNIST, ImageNet, or CIFAR-10. While these are useful resources, for true development to be made we need to be testing the tools used on these popular image collections on new sets of data, to evaluate both how useful and how generalisable the models are in their current state. These insights help the field to advance and allow us to apply the tools into many more fields, such as cultural data science. Therefore, much of this assignment looks at the process of how to collect a dataset and considers what problems can arise throughout this process. It then applies the collected dataset into an AlexNet CNN model to see how well it will perform. 

**Assignment Description**

This assignment collects 2 datasets of images, Banksy Streetart and Other Streetart, from Google images. It does this by scraping a list of image URLs using a short java-script, converts these images into images using a python script, removes duplicate images within each set using another python script and then creates a train and test set folder architecture. The structured datasets are then feed into an adapted AlexNet CNN model to see if the model can classify Banksy images from the other streetart. 

Purpose of the assignment: 
1. Create a reusable pipeline to collect and process image data from Google Images 
2. Consider what challenges we face using images collected in this raw sense 
3. Develop upon the CNN architecture by implementing an adapted binary CNN architecture inspired by the AlexNet model to help answer our research question 

Assignment task: 
Part 1: Develop the pipelines to collect the dataset 

Part 2: Pre-process the data (including removing duplicates and creating train-test splits) 

Part 3: Adapt the AlexNet model architecture to suit a binary classification question and run model 

## Scripts and Data 

**Data** 
The data consists of 2 sets collected from Google images using the following earch terms 


```
 Banksy                     | Other Street art                 
 -------------------------- | --------------------             
 “banksy art”               | “streetart -banksy”              
 “banksy graffiti”          | “swoon graffiti”                 
 “banksy political artwork” | “daze graffiti”                  
                            | “blek le rat graffiti”           
                            | “jean-michael basquiat graffiti” 
                            | “sheppard fairey graffiti”        
```
Details of this collection process can be found in the description on the Google_Image_Converter.py script. 

**Scripts** 
The assignment includes 4 scripts, which have been broken up into separate modules to enable them to stand as stand-alone tools for the process of collecting data and fitting that to the question at hand. These scripts are as follows: 


| Script | Description|
|--------|:-----------|
Google_Image_Converter.py| Converts image URLs to jpg files  
Duplicate_Removal.py | Removes duplicate images using numerical representation
Creating_Train_Test_Split.py | Splits the datasets into train and validation folders 
Banksy_CNN.py | Pre-processes data and runs it thorugh a CNN  


**Output**
The “out” folder contains a png file of the model architecture (AlexNext_model.png), a classification report (classification_report.txt), a plot of the model performance (model_history.png) and a summary of the model (model_summary.txt). 


## Methods 

The assignment took a 3-part methodological approach: 
1. Create generalisable pipelines to collect an image dataset 
2. Pre-process the data
3. Adapt the AlexNet CNN architecture into a binary style approach

__Part 1__
The initial data collection involved using a set of Javascript functions to gather the image URLs (this process is outlined in the 'js_console.js' script found in the Data Collection folder). This created a txt file of images from each Google search whereby every image URL was placed on a new line in the txt file. This file was then fed into the 'Google_Image_Converter.py' script which transformed the URLs into jpeg images and stored them in the dedicated folder. The script could take multiple txt files but needs to be run for each class of data, enabling it to fit into a multiclass situation also. The 'Duplicate_Removal.py' script was then run on each classes image directory to calculate whether duplicates exist and remove them if they do. Finally, the Creating_Train_Test_Split.py script was run to split the data into a parent training and validation folder, with a folder containing the jpg images for each class within. The four scripts used in this process have been collected into the data_collection folder along with the URL txt files scraped from Google. 

__Part 2__ 
The data was pre-processed in the Banksy_CNN.py script whereby the images were assigned a class label, split into new train and test sets, normalised by dividing by 255, and had their class labels binarised using sklearn's LabelBinarizer() function. 

__Part 3__ 
The data was then fed into an AlexNet inspired CNN architecture. The AlexNet is an award-winning architecture which builds upon the LeNet model by adding 3 extra layers. It was one of the first neural networks to start using the ReLU activation function, which it applies to each of its 8 layers, except the final output where a softmax activation is used. In this model's adaptation, the final softmax activation function is replaced with the sigmoid, to enable a binary classification. The 'adam' optimizer was used due to its flexible nature. 

The model's architecture took the following form: 

INPUT => CONV => ReLU => MAXPOOL => CONV => ReLU => MAXPOOL => CONV => RELU => CONV => RELU => CONV => RELU => MAXPOOL => 
FC => RELU => FC => RELU => OUTPUT => SIGMOID 

The model was then evaluated and visualised to create a classification report and model plot, which can be found in the output folder. 

## Operating the Scripts 

**Set-up**

__1. Clone the repository__ 

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository as classification_benchmarks_orlz
git clone https://github.com/Orlz/CDS_Visual_Analytics.git 

```


__2. Create the virtual environment__

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing:

```bash
$ source Computer_Vision04/bin/activate
```

**The scripts have been divided into 2 parts: The Data collection and The Modelling **

### The Data Collection

The data collection has been split into 4 modular scripts to allow easier reproducibility. Splitting them in this way and giving the scripts argparse command line functionality means each script can be easily adapted into a new context. (all scripts for data collection have been copied from src  into a separate folder called data_collection. The user can amend the paths to work the scripts from this folder if it is preferred) 

**Step One: Collecting the Image URLs**
To collect the image URLs, the user should follow these steps: 
1. Locate and open the js_console.js script 
(found in ../data_collection/Data_collection_scripts)

2. In a new Chrome Browser, navigate to a fresh Google Images search
(Can be done by typing: https://www.google.com/imghp?hl=EN) 

3. Type in the desired search term and then from the top left corner navigate through the following by clicking on each: 
‘View’ ==> ‘Developer’ ==> ‘JavaScript Console’
This will open a dark JavaScript console on your right-hand panel. 

4. Next, start scrolling down the page of Google images without clicking on any of the images. Continue scrolling until the images no longer become relevant to your search. 

5. When you are happy with the scroll, click into the js_console.js script and copy and paste each of the 4 functions into the right-hand JavaScript console, running each in turn. Finally, type in the function labelled “FUNCTION: DOWNLOAD URLs” and run it. This will download a txt file of image URLs, where each URL is on a new line. 

6. Repeat these steps for each search term, opening a new Google Images page each time

**Step 2: Convert the image URLSs to JPGs**
The next step is to convert the txt file of URLs into image JPG files in a new directory. This can be done by using the Google_Image_Converter.py script found in the src folder. 

The script has been set up to loop over all txt files found in the defined directory (args[“directory”]).  Therefore, the user just needs to place all the downloaded txt files relevant for each class into the correct category folder, use the argparse commands to suit their folder architecture, and run the script from the terminal. The script will need to be run for each class category. 

For example, the URL txt files for this project are found in data_collection/URL_txt_files where there is a folder for each of the classes, Banksy and Other Streetart. 

**Usage**
```bash
$ python3 src/Google_Image_Converter.py -o data/Images/Banksy_Streetart
```

**Step 3: Remove the Duplicates** 

Duplicate images should be removed before splitting the data into its train and test sets. This can be done by running the Duplicate_Removal.py script found in the src folder. 

Again, argparse arguments have been set up to allow the user to adapt the directory of the script so that it can run on all image directories required. There is also the option to run the script and either remove the duplicate photos immediately, or simply inform the user how many duplicates have been found without removing them. This is controlled using the “remove” argument. 

The script will need to be run on each class directory. For example, for this project the duplicate removal script was run on both the Banksy images (removing 60 images) and the Other Streetart images (removing 1 image). 

**Usage**
```bash
$ python3 src/Duplicate_Removal.py -r 1
```

**Step 4: Create Train and Test set folder architecture (Optional and not needed for this script)**

Once the data has been collected, we may want to arrange it into train and test set folders for use with other projects. To enable this, the Create_Train_Test_Split.py script was developed which can take any number of classes and create parent train and validation folders, with class image directories nested within. 

This is surprisingly complex but can now be easily completed for the dataset by employing the Creating_Train_Test_Split.py script found in the src folder. This way no folders need to be created manually. 

The script is a one-time use script and should not be attempted to be run multiple times, as it will throw error messages for directories that already exist. If problems arise in the first run, remove all newly created directories and start the script from scratch again. Argparse arguments allow the user to determine the size of the validation split and set up where the images should be pulled from (classes_dir) and saved to (root_dir). The path setup for this script is a little different than the other scripts and the user is recommended to read the description at the top of the script for optimal use. 

**Usage**
```bash
$ python3 src/Creating_Train_Test_Split.py --validation_size 0.2
```

## Modelling the CNN 

The image pre-processing, modelling, and evaluation of the model is all completed in one final script named Banksy_CNN.py. This can be found in the src folder. 

The script has a number of command line arguments which can be determined by the user, if desired. These are however optional: 

```
Letter call  | Parameter      | Required? | Input Type     | Description                                         
-----------  | -------------  |--------   | -------------  |                                                     
`-i`         | `--image_path` | No        | String         | Path to the full image directory                    
`-t`         | `--test_size`  | No        | Float          | Decimal number between 0 and 1 indicating test size 
`-n`         | `--n_epochs`   | No        | Integer        | The number of epochs to train the model on          
`-b`         | `--batch_size` | No        | Integer        | The size of the batch on which to train the model   
```

**Usage**
```bash
$ python3 src/Banksy_CNN.py --n_epochs 10
```

## Discussion of Results 

The model achieved acceptable accuracy, with a weighted and macro average of 74% (24% above chance). A slight difference was detected between the classes with Banksy images achieving a marginally higher accuracy (74%) compared to the other street art (73%). The below plot indicates the model has potential to improve this accuracy to around 80%, if run with 10 epochs instead. This is because after 10 epochs, we see the validation accuracy begins to decline (green) while the validation loss dramatically increases (red), suggesting the model is over-fitting the data after 10 epochs. This is further supported by the still declining training loss curve (blue).

![model_hist](https://github.com/Orlz/Detecting_Banksy/blob/main/out/model_history_20.png)

Achieving an accuracy above chance is definitely not exciting in the world of neural networks but it is perhaps surprising here, considering the images were fed in with a raw format and streetart images are typically complex and varying in style. It certainly supports Banksy’s persona and unique style – suggesting his work is indeed distinguishable from other great works of streetart. Nevertheless, it should be remembered that we are not clear on what features the model is picking up in the images, and accuracy rates such as these could be accredited to simple things such as the angle of the camera and zoom. This is quite likely in this context, where one could assume that images of Banksy art are taken from a closer angle and have less noise than perhaps a picture of a whole wall of streetart. Moreover, Banksy’s work is known for it’s simple colour schemes which are often greyscale with the occasional pop of colour. This is where running the model again on more processed data, after controlling for the zoom, colours, and orientation, could help to improve accuracy and validate the model. As the purpose of the assignment was to give the computer the same chance as a human, this has not been done for this context but is recommended for further explorations. Moreover, applying some visual feature mapping could provide critical insights into what the model is using to classify, which would guide the collection of a better dataset. 

___Please Note: The data folder contains only a limited subset of the data due to the computational limits of GitHub___
