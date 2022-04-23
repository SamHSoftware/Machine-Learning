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

### How to train your model. 

(1) Open ```RUNME_to_train_model.py```.

(2) Consider the following code...

```
# Function to train the ResNet. 
# Function input arg 1: num_epochs [int] --> The number of epochs to train the model for. 
# Function input arg 2: batch_size [int] --> The batch_size. 
# Function input arg 3: new_model [bool] --> When True, trains a new model. When False, trains a previous model which you select.
# Function input arg 4: display_plot [bool] --> When True, prints the plots of loss and accuracy. 
train_ResNet(num_epochs=10, 
             batch_size=5,
             new_model = True, 
             display_plot=True)
```

... and edit the input args as needed. 

(2) Run the code. This will create a GUI (the same as the one displayed above) asking you to select the folder of training data, namely ```training-data```, which we made earlier. The code will the train you ResNet. 

(3) The code will output the trained model (saved every 5 epochs using a custom callback), the loss and accuracy data (asa .csv file) and the graphs of loss and accuracy for each epoch. The loss graph will look something like this: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/ResNet-HistoNet/training-data/training-data-20220423_144006/loss_20220423_144006.png?raw=true" alt="The loss graph" width="500"/>

The accuracy isn't simple to calculate, because our data isn't categorical. Thus, a simple comfusion matrix isn't possible. Instead, I have made a custom accuracy function, which takes a parameter between 0 and 1. If the parameter, p, is 0.1, then the function will return ```1``` if each value of the predicted disttribution is within ```p``` of the corresponding ground truth value. Else, the function will return ```0```. With a batch, an average of these 1s and 0s can be calculated, to get a pseudo-accuracy value, which is a function of p. This is what the accuracy graph will look like, for multiple values of p:

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/ResNet-HistoNet/training-data/training-data-20220423_144006/accuracy_20220423_144006.png?raw=true" alt="The accuracy graph" width="500"/>

It looks like the model might be starting to overfit a little towards the end of the training, despite the regularization I added to the model. Thus, I've taken the trained from from epoch 20, as opposed to that of epoch 30. 

(4) The outputs listed above will be saved with date-time tags to a new folder named prediction-data-YYYYMMDD_hhmmss (that's the data time string at the end). 

## How to use the model to predict distributions. 

(1) Open and run ```RUNME_to_use_model.py```.

(2) A series of GUIs will appear (similar to those above). Each will have a title asking you to either select the trained model you wish to use (an ```.h5``` file) or the directory of images that you wish to produce graphs for. 

(3) A new directory will be created, called ```prediction-data-YYYYMMDD_hhmmss```. Inside, the distribution graphs will be stored. They will look like this: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/ResNet-HistoNet/training-data/prediction-data-20220423_155034/circle_area_image_3.png?raw=true" alt="A distribution of circle areas within an image" width="500"/>

The black bars represent the relative distribution, while the green bars represent the cumulative relative distribution. 