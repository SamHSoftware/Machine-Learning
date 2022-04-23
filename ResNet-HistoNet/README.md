# NB: This code is not yet complete.

# README for project: ResNet-HistoNet

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 
- This code is an exercise in the creation and optimization of ResNets. 
- Specifically, the ResNet is being used to preduct distribution data using KL-Divergence. 
- The code has the capacity to create it's own training data, a folder of images, each containing 9 circles of different volumes. It's these volumes (in pixels) that the code tries to predict. 

## Package architecture:
- ResNet-HistoNet:
    * This is where the main code is stored.
    * There are module files (these contain the individual functions) and RUNME files (these actually call (use) the functions).
    * There are .py files for general purpose IDEs, and .ipynb files for JupyterLab.
- img:
    * This is where images are stored. These images are used to illustrate this README document.
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

## How to use this code to generate the data needed for training. 

(1) Open and run ```RUNME_to_make_data.py```.

(2) This will create a GUI askign you to select a folder, within which a new directory of training data will be created. The new directory will be called ```training-data``` and the new GUI will look like this: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/ResNet-HistoNet/img/folder_selection.PNG?raw=true" alt="An example of the GUI used to select a folder" width="500"/>

(3) The images will be created within ```training-data```. Each image will contain 9 circles of different sizes. The distribution of circle volumes for each image is represented in a csv file within the same directory. The csv file is called ```area-data```. Each image has 1 row, where the first column isthe image path, and the proceeding columns are distribution values. 

### (1) Preparing your data for training. 

