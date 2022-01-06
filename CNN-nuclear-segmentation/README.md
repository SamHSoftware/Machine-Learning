# README for project: CNN-nuclear-segmentation

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 
- This code allows a user to train a pre-built [convolutional neural network (CNN)](https://en.wikipedia.org/wiki/CNN) to classify pixels within an image as being background, nuclear-border, nuclei or nuclear-debris.
- The example images that I have provided (those which will be used to train and test the CNN) is biological data. You can see bright spots in the images. These are the nuclei of human cells, genetically modified to fluoresce when illuminated with a laser. Specifically, these images were captured using a spinning disk confocal miscroscope. You can read more about these microscopes [here](https://en.wikipedia.org/wiki/Confocal_microscopy).
- Once trained, the CNN can be used to segment (numerically seperated from other pixel classes, e.g. background pixels) microscopy images of cell nuclei. 
- Alternatively, you could also use this model to classify different objects within images. It's up to you. 

## Requirements. 
(1) Please see the ```requirements.txt``` file (or ```requirements_conda.txt``` file if you are using anaconda) to note the packages (and their respective versions) which are needed for this code to run. 

## How to use this code to classify pixels into different classes. 

### (1) Preparing your data for training. 

First, you will need to present your model with training/testing data (in this case, grayscale images) and the corresponding ground truth data. The ground truth data needs to consist of pre-labelled images, in which each pixel value corresponds to a particular class. I have provided some example data to train and test the model. For the corresponding ground truth images, there are 3 different classes (4 if you count the background): 
- Pixels of value = 0 ... 'Background'  
- Pixels of value = 1 ... 'Nuclei'  
- Pixels of value = 2 ... 'Debris'  
- Pixels of value = 3 ... 'Nuclear-borders'  

Here is an example of a grayscale image: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/example_raw_image.png?raw=true" alt="grayscale image of cell nuclei and some surrounding nuclear debris" width="500"/>

Here is the corresponding ground truth labelled image: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/example_ground_truth.jpg?raw=true" alt="grayscale image of cell nuclei and some surrounding nuclear debris" width="500"/>

I created my ground truth labelled image using the MATLAB app, ```image labeller```. If you don't have MATLAB, then you can use [apeer.com](https://www.apeer.com/home/), which has a useful image labelling tool.  

Within the directory containing the training data, naming conventions are important. For each 'raw' image taken by the microscope (e.g. ```image_1.tif```), there needs to be an equivalent ground truth image with ''\_gtruth' appended onto the end (e.g. ```image_1_gtruth```). 

### (2) Selecting the data. 

Open, ```RUNME.py```. Here, you can select and run everything at all once if you're confident. If this is your first time, I'd recommend running the code piece by piece. First run the following code: 

```
the code for the directory gui
```

A GUI will appear (see the example below), with which you should select the folder containing your training dataset. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/folder_selection.PNG?raw=true" alt="An example of the GUI used to select training dataset directory" width="500"/>

### (3) Training your model.

Consider the following code: 

```
The code for the model. 
```

You will notice that there are many input arguments which allow you to specify what kind of ouputs get saved, how many epochs the model trains for, and whether the model saves itself for future bouts of training. These input args are all explained within the code above. 

When the model has trained, it will save a number of different image outputs to a new folder (named ```training data_YYYYMMDD_HHMMSS```) within the original training dataset directory. These ouputs contain the following:  

The training and validation loss:
<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/loss_graph.png?raw=true" alt="An example of the training and validation loss over several hundred epochs." width="500"/>

The training and validation accuracy:  
<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/CNN-nuclear-segmentation/img/accuracy_graph.png?raw=true" alt="An example of the training and validation accuracy over several hundred epochs." width="500"/>

The montage images (one per epoch). These can be indivudually inspected, or they can be (manually) made into a ```.gif``` to show how the model learns with each epoch. I've taken an excerpt from one such gif (minus the raw image, as that drove up the file size to push to github) to demonstrate this: 
 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Question_mark_%28black%29.svg/800px-Question_mark_%28black%29.svg.png" alt="An example of a gif which you can make to determine how your model trains over time." width="500"/>


### (4) Continuing the training for a previous model. 

Lets say you've trained a model for 200 epochs, but need to train it a little more. No problem. Within ```train_CNN``` set ```train_previous_model = True```. When the function is run, you will be prompted to select your training dataset (like last time) and then, you'll be prompted to select your old model (a file with the naming convention, ```multiclass_CNN_YYYYMMDD_HHMMSS.hdf5```). Your model will load in (alongside a ```.csv``` of the training data) and it will continue to train. 