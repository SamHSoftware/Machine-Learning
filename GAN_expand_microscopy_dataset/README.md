# README for project: GAN_expand_nuclear_datase

# *PLEASE NOTE: THIS PROJECT IS NOT YET COMPLETE. TRAINED MODEL EXAMPLES HAVE YET TO BE ADDED*

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 

## Package architecture:
- GAN_expand_nuclear_dataset:
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
(2) The code will train on your CPU. If you have set up CUDA and cuDNN to function with a NVIDIA GPU, that's fine, but you'll need to head into ```GAN_module.py``` (or ```GAN_modeule.ipynb```) and comment out the following line of code: 
```
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## How to use this code to classify pixels into different classes. 

### (1) Preparing your data for training. 
- This code assumed that you have a folder of images to be used to train the model. 
- The images in this folder need to be named in a specifc way. 
	* The first image is called ```image_1{file_extension}```. 
	* By extension, the n-th image is called ```image_n{file_extension}``` e.g., ```image_49.tif```.
- Other folder and files can exist within the training folder, but make sure there are only images of your training-image file type e.g. ```.tif```.
- The code assumes that the images are of specific dimensions. 
	* Height: 1024 pixels. 
	* Width: 1024 pixels.
	* Channels: 1 (the images should be greyscale, not RGB). 

### (2) Selecting the data. 

Open, ```RUNME.py```. Here, you can select and run everything at all once if you're confident with the variables your chosen. If this is your first time, I'd recommend running the code piece by piece. First run the following code: 

```
from GAN_module import * 

# A function to allow the user to select the folder contianing the data.
# Function inputs args: None. 
# Function output 1: The path of that the folder selected by the user. 
directory = folder_selection_dialog()
```

A GUI will appear (see the example below), with which you should select the folder containing your training dataset. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/GAN_expand_microscopy_dataset/img/folder_selection.PNG?raw=true" alt="An example of the GUI used to select training dataset directory" width="500"/>

### (3) Training your model.

Consider the following code: 

```
# Function to train the GAN and save the model outputs. 
# Function input arg 1: directory [string] --> The directory containing the training data.
# Function input arg 2: file_type [string] --> The file type of the training data e.g. '.tif'.
# Function input arg 3: save_data [bool] --> When True, saves the models and the training data.
# Function input arg 4: num_epochs [int] --> The number of epochs to train for. 
# Function input arg 5: batch_size [int, preferbly even] --> Num images processed per batch. NB: Your batch size can never equal more than 2*
# Function input arg 6: test_train_split [float] --> The fraction of the data which will be used for training. e.g. 0.8 would mean 80% be used for training, while 20% for testing.
# Function input arg 7: display_plot [bool] --> When True, prints the graphs of accuracy and loss.
# Function input arg 8: train_previous_model [bool] --> When true, the code will continue training a saved model of the user's choice. 
train_model(directory,
            file_type = '.tif',
            save_data = True,
            num_epochs = 5,
            batch_size = 4,
            test_train_split = 0.01,
            display_plot=True, 
            train_previous_model=False)
```

There are many input args here which need to be considered. Their explanations lie in the commented code above, and should be pretty self-explanatory.  

Once your code has finished running (and assuming that ```save_data = True```), several graphs will be saved to a directory containing information concerning the model's training. The directory will be called ```training_data_YYYYMMDD_HHMMSS```, where YYYYMMDD_HHMMSS refers to the timestamp when the function started running. 

These graphs include a record of loss and accuracy, both of which are derived from a dataset stored in ```training_log.csv```. There will also be graphical representations of model structures, as well as the models themselves, saved as ```generator_YYYYMMDD_HHMMSS.hdf5```, ```discriminator_YYYYMMDD_HHMMSS.hdf5``` and ```GAN_YYYYMMDD_HHMMSS.hdf5```. 

### (4) Continuing the training for a previous model. 

Continuing the training for a previous model is simple. Run the ```train_model``` function with ```train_previous_model = True```, and a GUI will appear. With this GUI select the folder containing the previously trained models. The code will create a new training folder, and will append new training data onto a new (copied and pasted) version of ```training_log.csv```. 

### (5) Use your trained model to generate fake microscopy images. 

Use the following code: 

```
from GAN_module import * 

# Function to use a trained generator to create our fake data. 
# Function input arg 1: number_of_images [int] --> The number of images you wish to generate. 
use_generator(number_of_images)
```

The first step is the replacement of ```number_of_images``` with an integer, which will determine the number of fake images you generate. 

The images will be created within a new directory: ```training_data_YYYYMMDD_HHMMSS\\fake_images```. The images will be sclaed between 0 and 255 and will be of equivalent size to the training images. The first image will be called ```image_{0}.png```, with the second being called ```image_{1}.png``` etc.. 