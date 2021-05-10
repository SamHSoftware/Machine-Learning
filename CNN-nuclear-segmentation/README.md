# README for project: CNN-nuclear-segmentation

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 
- This code allows a user to train a pre-built [convolutional neural network (CNN)](https://en.wikipedia.org/wiki/CNN) to classify pixels within an image as being background, nuclear-border, nuclei or nuclear-debris.
- The example images that I have provided (those which will be used to train and test the CNN) is biological data. You can see bright spots in the images. These are the nuclei of human cells, genetically modified to fluoresce when illuminated with a laser. Specifically, these images were captured using a spinning disk confocal miscroscope. You can read more about these microscopes [here](https://en.wikipedia.org/wiki/Confocal_microscopy).
- Once trained, the CNN can be used to segment (numerically seperated from other pixel classes, e.g. background pixels) microscopy images of cell nuclei. 
- Alternatively, you could also use this model to classify differnt objects within images. It's up to you. 

## Requirements. 
(1) Please see the ```requirements.txt``` file (or ```requirements_conda.txt``` file if you are using anaconda) to note the packages (and their respective versions) which are needed for this code to run. 

## How to use this code to classify pixels into different classes. 

(1) First, you will need to present your model with training/testing data (in this case, grayscale images) and the corresponding ground truth data. The ground truth data needs to consist of pre-labelled images, in which each pixel value corresponds to a particular class. I have provided some example data to train and test the model. For the corresponding ground truth images, there are 3 different classes (4 if you count the background): 
- Pixels of value = 0 ... 'Background'  
- Pixels of value = 1 ... 'Nuclei'  
- Pixels of value = 2 ... 'Debris'  
- Pixels of value = 3 ... 'Nuclear-borders'  