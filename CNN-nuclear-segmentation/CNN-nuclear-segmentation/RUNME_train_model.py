# Import the necessary packages.
from CNN_nuclei_module import *

# A function to allow the user to select the folder contianing the data.
# Function inputs args: None. 
# Function output 1: The path of that the folder selected by the user. 
directory = folder_selection_dialog()

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