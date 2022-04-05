# README for project: CNN-nuclear-segmentation

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 
- This code allows a user to train a pre-built [convolutional neural network (CNN)](https://en.wikipedia.org/wiki/CNN) to classify pixels within an image as being background, nuclear-border, nuclei or nuclear-debris.
- The example images that I have provided (those which will be used to train and test the CNN) is biological data. You can see bright spots in the images. These are the nuclei of human cells, genetically modified to fluoresce when illuminated with a laser. Specifically, these images were captured using a spinning disk confocal miscroscope. You can read more about these microscopes [here](https://en.wikipedia.org/wiki/Confocal_microscopy).
- Once trained, the CNN can be used to segment (numerically seperated from other pixel classes, e.g. background pixels) microscopy images of cell nuclei. 
- Alternatively, you could also use this model to classify different objects within images. It's up to you. 

## Package architecture:
- CNN-nuclear-segmentation:
    * This is where the main code is stored.
    * There are module files (these contain the individual functions) and RUNME files (these actually call (use) the functions).
    * There are .py files for general purpose IDEs, and .ipynb files for JupyterLab.
- img:
    * This is where images are stored. These images are used to illustrate this README document.
- .gitignore:
    * I use software called 'Git Bash' to sync my local files with those within this GitHub repositry. The .gitignore file contains a list of directories (e.g. text files with project notes), and prevents Git Bash from uplodaing them; I don't want to clutter up this repo!
- LICENCE.txt:
    * The licence explaining how this code can be used.
- README.md:
    * The file which creates the README for this code.
- environment.yml:
    * A file to allow you to re-create this code's environment in conda.
- requirements.txt:
    * A file to allow you to re-create this code's environment using pip.

## Requirements. 
(1) Please see the ```requirements.txt``` file (or ```environment.yml``` file if you are using conda) to note the packages (and their respective versions) which are needed for this code to run. 

## How to use this code to classify pixels into different classes. 

### (1) Preparing your data for training. 

First, you will need to present your model with training/testing data (in this case, grayscale images) and the corresponding ground truth data. The ground truth data needs to consist of pre-labelled images, in which each pixel value corresponds to a particular class. I have provided some example data to train and test the model. For the corresponding ground truth images, there are 3 different classes (4 if you count the background): 
- Pixels of value = 0 ... 'Background'  
- Pixels of value = 1 ... 'Nuclei'  
- Pixels of value = 2 ... 'Debris'  
- Pixels of value = 3 ... 'Nuclear-borders'  

Here is an example of a grayscale image (left) and the corresponding ground truth labelled image (right):

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/example_raw_image.png?raw=true" alt="grayscale image of cell nuclei and some surrounding nuclear debris" width="300"/> <img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/example_ground_truth.jpg?raw=true" alt="grayscale image of cell nuclei and some surrounding nuclear debris" width="300"/>

I created my ground truth labelled image using the MATLAB app, ```image labeller```. If you don't have MATLAB, then you can use [apeer.com](https://www.apeer.com/home/), which has a useful image labelling tool.  

Within the directory containing the training data, naming conventions are important. For each 'raw' image taken by the microscope (e.g. ```image_1.tif```), there needs to be an equivalent ground truth image with ''\_gtruth' appended onto the end (e.g. ```image_1_gtruth```).  

Image dimensions are also important. This model expects images to be 2560 pixels wide, and 2160 pixels high. 

### (2) Selecting the data. 

Open, ```RUNME.py```. Here, you can select and run everything at all once if you're confident. If this is your first time, I'd recommend running the code piece by piece. First run the following code: 

```
# Import the necessary packages.
from CNN_nuclei_module import *

# A function to allow the user to select the folder contianing the data.
# Function inputs args: None. 
# Function output 1: The path of that the folder selected by the user. 
directory = folder_selection_dialog()
```

A GUI will appear (see the example below), with which you should select the folder containing your training dataset. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/folder_selection.PNG?raw=true" alt="An example of the GUI used to select training dataset directory" width="500"/>

### (3) Training your model.

Consider the following code: 

```
# A function capable of training a CNN to classifying pixels within .tif microscopy images of cell nuclei. 
# Function input 1: directory [str] --> The directory containing the original and gtruth data. 
# Function input 2: save_plot [bool] --> When True, graphical data will be saved. 
# Function input 3: display_plot [bool] --> When True, graphical data will be displayed in the console. 
# Function input 4: save_model [bool] --> When True, saves the model to the directory containing the training data. 
# Function input 5: train_previous_model [bool] --> When True, the user is prompted to select a previously trained model, in order to continue it's training.
# Function input 6: num_epochs [int] --> The number of epochs to train the model.
# Function input 7: make_movie [bool] --> When true, outputs a movie of the model learning over time. 
train_CNN(directory,
          save_plot=True,
          display_plot=True,
          save_model=True, 
          train_previous_model=False,
          num_epochs=300, 
          make_the_movie=True)
```

You will notice that there are many input arguments which allow you to specify what kind of ouputs get saved, how many epochs the model trains for, and whether the model saves itself for future bouts of training. These input args are all explained within the code above. 

When the model has trained, it will save a number of different image outputs to a new folder (named ```training data_YYYYMMDD_HHMMSS```) within the original training dataset directory. These ouputs contain the following:  

The training and validation loss (left) and the training and validation accuracy (right):
<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/loss_graph.png?raw=true" alt="An example of the training and validation loss over several hundred epochs." width="500"/><img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/accuracy_graph.png?raw=true" alt="An example of the training and validation accuracy over several hundred epochs." width="500"/>

The montage images (one per epoch). These can be indivudually inspected, or they can be (manually) made into a ```.gif``` to show how the model learns with each epoch. I've taken an excerpt *(this is **not** the full training process)* from one such gif *(minus the raw image, as that drove up the file size to push to github)* to demonstrate this: 
 
<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/animation.gif?raw=true" alt="An example of a gif which you can make to determine how your model trains over time." width="800"/>

### (4) Continuing the training for a previous model. 

Lets say you've trained a model for 200 epochs, but need to train it a little more. No problem. Within ```train_CNN``` set ```train_previous_model = True```. When the function is run, you will be prompted to select your training dataset (like last time) and then, you'll be prompted to select your old model (a file with the naming convention, ```multiclass_CNN_YYYYMMDD_HHMMSS.hdf5```). Your model will load in (alongside a ```.csv``` of the training data) and it will continue to train. 

### (5) Use your trained model to classify the pixels of images within a folder. 

Use the following code: 

```
# Function to use a trained CNN to classify data and save the results in a new directory.
use_CNN()
```

Two pop-up GUIs will appear. With the first, select the trained model that you wish to use. With the second, select the directory of images that you need to process. For simplicities sake, I've assumed that the folder of greyscale images, contains only that, images which need to be (and can be) processed by the model.    
The processed images will appear in a new directory (```classified_images```), which in turn will reside within the folder containing the unprocessed images. 