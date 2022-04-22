from code_module import *

# Function to train the ResNet. 
# Function input arg 1: num_epochs [int] --> The number of epochs to train the model for. 
# Function input arg 2: batch_size [int] --> The batch_size. 
# Function input arg 3: new_model [bool] --> When True, trains a new model. When False, trains a previous model which you select.
# Function input arg 4: display_plot [bool] --> When True, prints the plots of loss and accuracy. 
train_ResNet(num_epochs=10, 
             batch_size=5,
             new_model = True, 
             display_plot=True)