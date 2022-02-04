from GAN_module import * 

# A function to allow the user to select the folder contianing the data.
# Function inputs args: None. 
# Function output 1: The path of that the folder selected by the user. 
directory = folder_selection_dialog()

# Function to train the GAN and save the model outputs. 
# Function input arg 1: directory [string] --> The directory containing the training data.
# Function input arg 2: file_type [string] --> The file type of the training data e.g. '.tif'.
# Function input arg 3: save_data [bool] --> When True, saves the models and the training data.
# Function input arg 4: num_epochs [int] --> The number of epochs to train for. 
# Function input arg 5: batch_size [int, preferbly even] --> Num images processed per batch. NB: Your batch size can never equal more than 2*
# Function input arg 6: test_train_split [float] --> The fraction of the data which will be used for training. e.g. 0.8 would mean 80% be used for training, while 20% for testing.
# Function input arg 7: display_plot [bool] --> When True, prints the graphs of accuracy and loss.
train_model(directory,
            file_type = '.tif',
            save_data = True,
            num_epochs = 25,
            batch_size = 6,
            test_train_split = 0.1,
            display_plot=True)