from tkinter import *
from tkinter import filedialog
import time 
from functools import wraps
import os
import numpy as np
from math import floor 
from PIL import Image
import cv2 
import matplotlib.pyplot as plt 
import random
import math 
import pandas as pd
from tqdm import trange
from pandas import read_csv
from matplotlib.ticker import MaxNLocator
import keras 
from keras.layers import Input, BatchNormalization, Activation, Conv2D, Dropout, Add, MaxPooling2D, Dense, Flatten, Lambda
from tensorflow.keras import regularizers
import tensorflow as tf 
from keras.models import Model
from keras import backend as K 
from datetime import datetime 
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Function to allow the user to select the folder contianing the data.
# Function inputs arg 1: message [string] --> The title for the GUI. 
# Function output 1: directory [string] --> The path of that the folder selected by the user. 
def select_folder(message):
    root = Tk()
    root.title(message)
    root.filename = filedialog.askdirectory(initialdir="/", title=message)
    directory = root.filename
    root.destroy()

    return directory

# Function to allow the user to select the file they need.
# Function inputs arg 1: message [string] --> The title for the GUI. 
# Function output 1: file_directory [string] --> The path of that the folder selected by the user. 
def select_file(message):
    root = Tk()
    root.title(message)
    root.filename = filedialog.askopenfilename(initialdir="/", title=message)
    file_path = root.filename 
    root.destroy()
    
    return file_path

# Decorator in-case I need to optimise code from a temporal point of view. 
# Decorator input arg 1: f --> The function to be decorated. 
# Decorator output: timed (string) --> Time taken by decorated function to run (seconds). 
def timing(f): 
        
    # This ensures that the function metadata (e.g. name, docstring,
    # etc.) isn't overwritten when we use the decorator.
    @wraps(f) 
    
    def with_timing(*args, **kw):
        
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        
        print(f'Time taken (s): {end_time - start_time}')
        
        return result
    return timing   

# Fucntion to creata a training dataset of images with circles of different areas, and .csv files wth the corresponding area information. 
# Function input arg 1: num_images [int] --> The number of trainig images you wish to create. 
def create_training_data(num_images): 
    
    # Select the folder within which you want to create the training dataset directory: 'training-data'.
    parent_dir = select_folder('Select the folder within which "training-data" will be made')
    
    # Create the training-data directory. 
    if not os.path.exists(os.path.join(parent_dir, "training-data")):
        os.makedirs(os.path.join(parent_dir, 'training-data'))        
    
    # Create and pandas DataFrame to log away circle area data. 
    i_data = pd.DataFrame(columns=['imagePath', '10 < x <= 340', '340 < x <= 1000', '1000 < x <= 1700', '1700 < x <= 2500', '2500 < x <= 50000' ])
    
    # Iteratively produce the training images. 
    for i in (x := trange(num_images)):
        # Set the description for the trange progress bar. 
        x.set_description(f"Creating training image:{str(i+1)}")
        
        # First create a blank canvas up 0-value pixels. 
        canvas = np.zeros((256,256))
        
        # Randomly generate param, such that param can influence the area of the circles. 
        rand_param = random.randrange(-10, +10, 1)
        
        # Create an empty list to store the j_list data. 
        j_list = []
        
        # Create 9 circles within the image. 
        for j in range(9):

            # Determine the column and row index. 
            col = ((j%3) + 1) * 64
            row  = (floor(j/3) + 1) * 64
            
            # Get a random radius. 
            radius = random.randrange(12, 25, 1) + rand_param
            
            # Set the circle parameters. 
            center = (col, row)
            axes = (radius, radius)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (1)
            thickness = -1

            # Add the elipse to the image.
            cv2.ellipse(canvas, center, axes, angle, startAngle, endAngle, color, thickness)
            
            # Take note of the area of the circle added to the image.
            area = math.pi * radius**2
            j_list.append(area)
        
        # Tally the area sizes into bins.
        bins = [10, 340, 1000, 1700, 2500, 50000]
        tally = np.ndarray.tolist(np.digitize(np.array(j_list), bins, right=True))
        tally_2 = []
        for o in range(1, len(bins)):
            value = tally.count(o)
            tally_2.append(value)
        
        # Scale the distribution such that it sums to 1. 
        tally_2 = np.ndarray.tolist(np.array(tally_2) / sum(np.array(tally_2)))
        
        # Determine the path for the new image. 
        image_path = os.path.join(parent_dir, 'training-data', f'image_{i}.tif')

        # Append the tally to the image path. 
        i_list = []
        i_list.append(image_path)
        i_list.extend(tally_2)

        # Add the list as a new row to the pandas DataFrame. 
        i_data.loc[len(i_data)] = i_list

        # Save the image. 
        img = Image.fromarray(canvas)
        img.save(image_path)
        
    # Save the i_data DataFrame. 
    csv_path = os.path.join(parent_dir, 'training-data', 'area_data.csv')
    i_data.to_csv(csv_path, index=False)
    
# Function to load in a list of training image paths and their respective distributions. 
# Function input arg 1: directory [string] --> The directory of training data. 
# Function output 1: image_paths [list] --> The list of image paths. 
# Function output 2: distributions [numpy array] --> The distributions.
def get_names_and_distributions(directory):
    
    # First, load in the csv file. 
    csv_path = os.path.join(directory, 'area_data.csv')
    csv_data = read_csv(csv_path)
    
    # Seperate out the data. 
    image_paths = [_ for _ in csv_data.iloc[:,0]]
    distributions = csv_data.iloc[:,1:6].to_numpy()
    
    return image_paths, distributions

# Function to display and save the training loss and validation loss per epoch.
# Function input arg 1: training_loss --> Array of size 1 x num_epochs. This array contains the calculated values of loss for training. 
# Function input arg 2: validation_loss --> Array of size 1 x num_epochs. This array contains the calculated values of loss for validation. 
# Function input arg 3: display_plot --> True or Flase. When true, displays the plot. 
# Function input arg 4: directory --> The directory containing the training dataset. 
# Function input arg 5: date_time --> The datetime string in the format of 'YMD_HMS'. 
def loss_graph(training_loss, 
               validation_loss, 
               display_plot,
               directory, 
               date_time):
    
    # Plot the loss per epoch. 
    y = list(range(0,len(training_loss)))
    plt.plot(y, training_loss, label = "Training loss")
    plt.plot(y, validation_loss, label = "Validation loss")
    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Loss', labelpad=10) # The labelpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot.
    folder_name = f'training-data-{date_time}'
    if not os.path.exists(os.path.join(directory, folder_name)):
        os.makedirs(os.path.join(directory, folder_name))
    file_path = os.path.join(directory, folder_name, f'loss_{date_time}.png')
    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# Function to display the accuary with each training epoch. 
# Function input arg 1: acc_01 [Array (1 x num_epochs)] --> The training acc with p = 0.1. 
# Function input arg 2: val_acc_01 [Array (1 x num_epochs)] --> The validation acc with p = 0.1. 
# Function input arg 3: acc_02 [Array (1 x num_epochs)] --> The validation acc with p = 0.2. 
# Function input arg 4: val_acc_02 [Array (1 x num_epochs)] --> The validation acc with p = 0.2. 
# Function input arg 5: acc_03 [Array (1 x num_epochs)] --> The validation acc with p = 0.3. 
# Function input arg 6: val_acc_03 [Array (1 x num_epochs)] --> The validation acc with p = 0.3. 
# Function input arg 7: date_time [str] --> The datetime string to add file names. Format: YYYYMMDD_HHMMSS.
# Function input arg 8: display_plot [bool] --> True or False. When True, displays the plot while the code is running. 
# Function input arg 9: directory [str] --> The path to the training_directory.
def create_accuracy_graph(acc_01, 
                          val_acc_01, 
                          acc_02, 
                          val_acc_02, 
                          acc_03, 
                          val_acc_03, 
                          date_time,
                          display_plot,
                          directory):
    
    # Create the x data. 
    x = list(range(0,len(acc_01)))
    
    # Create the accruary plot.
    plt.plot(x, acc_01, label="Training accuracy, p=0.1")
    plt.plot(x, val_acc_01, label="Validation accuracy, p=0.1")
    plt.plot(x, acc_02, label="Training accuracy, p=0.2")
    plt.plot(x, val_acc_02, label="Validation accuracy, p=0.2")
    plt.plot(x, acc_03, label="Training accuracy, p=0.3")
    plt.plot(x, val_acc_03, label="Validation accuracy, p=0.3")
    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Accuracy', labelpad=11) 
    plt.xlabel('Epoch', labelpad=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0,1)
    
    # Save the plot. 
    
    # If the training_data folder doesn't already exist, create it. 
    folder_name = f'training-data-{date_time}'
    if not os.path.exists(os.path.join(directory, folder_name)):
        os.makedirs(os.path.join(directory, folder_name))

    # Save the plot. 
    file_path = os.path.join(directory, folder_name, f'accuracy_{date_time}.png')
    plt.savefig(file_path, dpi=250, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()  
        
# Function to create a graph of circle area distribution data. 
# Function input arg 1: distribution [array] --> The (n,) distribution array.
# Function input arg 2: directory [str] --> When training the model, the training_data path. When testing the model, the testing_data path.
# Function input arg 3: display_plot [bool] --> True or False. When True, displays the plot while the code is running. 
# Function input arg 4: date_time [str] --> The datetime string to label the saves graphs. Formatted as YYYYMMDD_HHMMSS.
# Function input arg 5: name_tag [str] --> A string which will be added to the file names. 
def make_distribution_graph(distribution,
                            directory,
                            display_plot,
                            date_time,
                            name_tag='training_'):
    
    # Create the x data labels.
    x = ('10<x<=340', '340<x<=1000', '1000<x<=1700', '1700<x<=2500', '2500<x')
    
    # Create the x data locations. 
    x_loc = [0, 1, 2, 3, 4]
    
    # Calculate the cumulative distribution 
    c_distribution = []
    for j in range(len(distribution)):
        array_section = np.array(distribution[0:j+1])
        array_sum = sum(array_section)
        c_distribution.append(array_sum)
    
    # Create the graph.
    fig, axis_1 = plt.subplots()
    axis_2 = axis_1.twinx()
    axis_1.bar(x, c_distribution, color = "green")
    axis_2.bar(x, distribution, color = "black")
    axis_1.set_xlabel('Circle area (pixels)', fontsize = 15, labelpad=10)
    axis_1.set_ylabel('Proportion of circle\nareas', color='black', fontsize = 15, labelpad=10)
    axis_2.set_ylabel('Cumulative proportion of\ncircle areas', color='green', fontsize = 15, labelpad=10)
    axis_1.set_ylim([0, 1.1])
    axis_2.set_ylim([0, 1.1])
    axis_1.set_xticks(x_loc)
    axis_1.set_xticklabels(x, rotation=-45, ha='left', rotation_mode='anchor')
    
    # Save the plot if the user desires it.

    # If the directory doesn't exist, make it. 
    folder_name = f'prediction-data-{date_time}'
    if not os.path.exists(os.path.join(directory, folder_name)):
        os.makedirs(os.path.join(directory, folder_name))

    # Save the graph.
    file_path = os.path.join(directory, folder_name, f'circle_area_{name_tag}.png')
    plt.savefig(file_path, dpi=250, bbox_inches='tight')
    
    # Show the plot with matplotlib if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# Generator class to load in batches of data one by one. 
class Custom_Generator(keras.utils.all_utils.Sequence) :
    
    # Initialize the object. 
    def __init__(self, image_paths, distributions, batch_size, list_idxs, num_channels):
        self.image_paths = image_paths
        self.distributions = distributions 
        self.batch_size = batch_size 
        self.list_idxs = list_idxs
        self.num_images = len(self.list_idxs)
        self.num_channels = num_channels
        self.indexes = np.arange(self.num_images)

    # At the start of each epoch, generate a list of indexes and shuffle them such that the batches aren't identical between epochs. 
    def on_epoch_start(self):
        self.indexes = np.random.shuffle(self.indexes) 
        
    # Calculate the number of batches we need.
    def __len__(self):

        # Flooring prevents empty batches. 
        return int(math.floor((len(self.image_paths) / float(self.batch_size))))
    
    # Create one batch of data, where index is the batch number. 
    def __getitem__(self, index):

        # Determine which of our indexes we can use.
        start = index*self.batch_size
        if (index+1)*self.batch_size > self.num_images:
            end = self.num_images
        end = (index+1)*self.batch_size
        
        indexes = self.indexes[start : end]
        
        # Iteratively construct the x (image) batch. 
        for i in indexes: 
            img = cv2.imread(self.image_paths[i], -1)
            img = (img - img.min()) / (img.max() - img.min()) 
            img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
            if indexes[0] - i == 0: 
                x = img
            else: 
                x = np.append(x, img, axis=0)
            
        # Construct the y (distribution) batch. 
        for i in indexes: 
            distribution = self.distributions[i]
            distribution = distribution / sum(distribution)
            distribution = np.reshape(distribution, (1, distribution.shape[0]))
            if indexes[0] - i == 0: 
                y = distribution
            else: 
                y = np.append(y, distribution, axis=0)
        
        return x, y
    
# Function to create a resusuable identity block (a.k.a. a residual block) for the ResNet. 
# Function input arg 1: input_layer [array, float 16] --> The input layer. 
# Function input arg 2: output_filters [int] --> The number of filters passed over the output.
# Function output 1: output [array, float 16] --> The output layer. 
def identity_block(input_layer,
                   output_filters):
    
    # Duplicate the input layer such that it can be used within the skip connection. 
    input_layer_2 = input_layer

    # Perform convolution 1.
    conv_1 = Conv2D(output_filters, 
                    kernel_size=(1,1), 
                    padding='same', 
                    kernel_initializer='he_normal', 
                    kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                    bias_regularizer=regularizers.L2(1e-7),
                    activity_regularizer=regularizers.L2(1e-7))(input_layer_2)
    batchNorm_1 = BatchNormalization(axis=3)(conv_1)
    relu_1 = Activation('relu')(batchNorm_1)

    # Perform convolution 2.
    conv_2 = Conv2D(output_filters, 
                     kernel_size=(1,1), 
                     padding='same', 
                     kernel_initializer='he_normal',  
                     kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                     bias_regularizer=regularizers.L2(1e-7),
                     activity_regularizer=regularizers.L2(1e-7))(relu_1)
    batchNorm_2 = BatchNormalization(axis=3)(conv_2)
    relu_2 = Activation('relu')(batchNorm_2)

    # Perform convolution 3. 
    conv_3 = Conv2D(output_filters, 
                    kernel_size=(1,1), 
                    padding='same', 
                    kernel_initializer='he_normal',  
                    kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                    bias_regularizer=regularizers.L2(1e-7),
                    activity_regularizer=regularizers.L2(1e-7))(relu_2)
    batchNorm_3 = BatchNormalization(axis=3)(conv_3)

    # Add the skip connection.
    output = Add()([batchNorm_3, input_layer])
    
    # Add the activation function. 
    output = Activation('relu')(output)
    
    # Add dropout. 
    output = Dropout(0.1)(output)
    
    return output

# Function to create a reusuable convolutional block. 
# Function input arg 1: input_layer [array, float 16] --> The input layer. 
# Function input arg 2: stride [int] --> The stride of the second conv layer. When stride > 1, the image will be pooled. 
# Function input arg 3: output_filters [int] --> The number of filters passed over the output.  
# Function output 1: output_layer [array, float 16] --> The output. 
def convolutional_block(input_layer,
                        stride,
                        output_filters):
    
    # Perform the first convolution
    conv_1 = Conv2D(output_filters, 
                    kernel_size=(1,1), 
                    padding='same', 
                    kernel_initializer='he_normal', 
                    kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                    bias_regularizer=regularizers.L2(1e-7),
                    activity_regularizer=regularizers.L2(1e-7))(input_layer)
    batchNorm_1 = BatchNormalization(axis=3)(conv_1)
    relu_1 = Activation('relu')(batchNorm_1)
    
    # Perform the second convolution.
    conv_2 = Conv2D(output_filters, 
                    kernel_size=(3,3), 
                    strides=(stride,stride), 
                    padding='same', 
                    kernel_initializer='he_normal',  
                    kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                    bias_regularizer=regularizers.L2(1e-7),
                    activity_regularizer=regularizers.L2(1e-7))(relu_1)
    batchNorm_2 = BatchNormalization(axis=3)(conv_2)
    relu_2 = Activation('relu')(batchNorm_2)
    
    # Perform the third convolution. 
    conv_3 = Conv2D(output_filters, 
                    kernel_size=(1,1), padding='same', 
                    kernel_initializer='he_normal',  
                    kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                    bias_regularizer=regularizers.L2(1e-7),
                    activity_regularizer=regularizers.L2(1e-7))(relu_2)
    batchNorm_3 = BatchNormalization(axis=3)(conv_3)

    # Ensure the dimensions are correct.
    if  (output_filters == input_layer.shape[-1]) and (batchNorm_3.shape[0] == input_layer.shape[0]) and (batchNorm_3.shape[1] == input_layer.shape[1]):
        input_layer_2 = input_layer
    else:
        input_layer_2 = Conv2D(output_filters, 
                               kernel_size=(1,1), 
                               strides=(stride,stride), 
                               padding='same', 
                               kernel_initializer='he_normal',  
                               kernel_regularizer=regularizers.L1L2(l1=1e-7, l2=1e-7),
                               bias_regularizer=regularizers.L2(1e-7),
                               activity_regularizer=regularizers.L2(1e-7))(input_layer)
        input_layer_2 = BatchNormalization(axis=3)(input_layer_2)
    
    # Add the skip connection.
    output = Add()([batchNorm_3, input_layer_2])
    
    # Add the activation function. 
    output = Activation('relu')(output)
    
    # Add dropout. 
    output = Dropout(0.1)(output)
    
    return output

# Function to create a ResNet. 
# Function input arg 1: input_height [int] --> The height of the input (pixels). 
# Function input arg 2: input_width [int] --> The width of the image in (pixels). 
# Function input arg 3: input_channels [int] --> The number of channels. For this project, the inputs have a single channel. 
def create_resnet(input_height,
                  input_width,
                  input_channels): 
    
    #####################
    # Process the inputs.
    #####################
    
    # Define the input dimensions.
    inputs = Input((input_height, input_width, input_channels))
    #print("inputs:", inputs.shape)
    
    # Perform an initial 7x7 convolution to detect features within the input images.
    X = Conv2D(64, (7,7), strides=(2,2), kernel_initializer='he_normal', padding='same')(inputs)
    
    # Batch normalization is used to tackle 'internal covariate shift'. This describes a situation in which inputs are broadly distributed, and seem to change with each batch.
    # This can cause the models parameters (especilly those in deep layers) to constantly chase a moving target. 
    # To help normalize the means and variance of these inputs, batch norm layers are used. 
    X = BatchNormalization(axis=3)(X)
    
    # Apply an activation function to the sum of weighted inputs.
    X = Activation('relu')(X)
    
    # Max pooling will downscale the image, but will also identify the most prevalent features within the feature maps. 
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    
    ################################################################
    # Pass the previous layers to convolutional and identity blocks.
    ################################################################
    
    # Increase the number of filters to 128.
    X = convolutional_block(input_layer=X, stride=1, output_filters=128)
    X = identity_block(input_layer=X, output_filters=128)
    X = identity_block(input_layer=X, output_filters=128)
    
    # Reduce the image size to 32x32 and increase the number of filters to 256. 
    X = convolutional_block(input_layer=X, stride=2, output_filters=256)
    X = identity_block(input_layer=X, output_filters=256)
    X = identity_block(input_layer=X, output_filters=256)
    
    # Reduce the image size to 16x16. 
    X = convolutional_block(input_layer=X, stride=2, output_filters=256)
    
    ################################################################
    # Flatten the feature maps to produce an un-scaled distribution.
    ################################################################
    
    # Conv2D layerto reduce the number of filters, followed by ReLu.
    X = Conv2D(128, (3,3), kernel_initializer='he_normal', padding='same')(X)
    X = Activation('relu')(X)
    
    # Conv2d layer to reduce the number of filters, followed by ReLu.
    X = Conv2D(64, (1,1), kernel_initializer='he_normal', padding='same')(X)
    X = Activation('relu')(X)
    
    # Flatten the image and add dropout.
    X = Flatten()(X)
    X = Dropout(0.1)(X)
    X = Dense(5)(X)
    
    # A ReLu activation will make sure that our outputs are positive.
    X = Activation('relu')(X)
    
    ###########################################################################
    # Scale the distribution. Divide each value by the sum of the distribution. 
    ###########################################################################
    
    outputs = Lambda(lambda X: X / tf.keras.backend.sum(X, axis=1)[:,None])(X)
    
    ###########################################
    # Finish defining model inputs and outputs.
    ###########################################
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Function to log a pseudo-accuracy. Returns 1 if each value of pred distribution is within p of the corresponding value of the gtruth distribution.
# Fucntion input arg 1: p [float] -- > Between 0 and 1. The value described above.
# Function output 1: function --> The object. 
def custom_accuracy(p):
    def fn(y_true, y_pred):

        # Keras expects data to be symbolic, thus, normal numpy cannot be used. 
        # Instead, use keras backend functions. 
        
        # Calculate absolute differences between the distributions. 
        differences = abs(y_true - y_pred)

        # Get a boolean representation of the distributions where a difference is less than p. 
        differences = (differences < p)
        differences = K.all(differences, axis=1) 

        # Take the mean of our boolean tensor. 
        mean_accuracy = K.mean(differences)

        return mean_accuracy

    fn_name = str(p).replace(".", "")
    fn.__name__ = 'acc_{}'.format(fn_name)

    return fn

# Callback to save the model at regular intervals during training. 
# Class input 1: interval [int] --> The model will be saved each interval epochs. 
# Class input 2: directory [string] --> The training directory, as selected by the user early on.
# Class input 3: date_time [string] --> The date time string noted when the model is first run. Format: YYYYMMDD_hhmmss.
# Class method output 1: model --> The saved keras model. Format: ResNet_YYYYMMDD_hhmmss_Epoch_{epoch}. 
class SaveModelRegularly(tf.keras.callbacks.Callback):
    
    # Initialise class attributes. 
    def __init__(self, interval, directory, date_time):
        super().__init__() # This allows us to access the methods of the parent class (the tf callback).
        self.interval = interval 
        self.directory = directory 
        self.date_time = date_time 
        
    # At the end of the designated epochs, save our model. 
    def on_epoch_end(self, epoch, logs=None):
        if self.interval > 0 and epoch%self.interval == 0:
            
            # We need a directory within which we can save our model.
            # If it hasn't been created, create it. 
            folder_name = f'training-data-{self.date_time}'
            if not os.path.exists(os.path.join(self.directory, folder_name)):
                os.makedirs(os.path.join(self.directory, folder_name))

            # Save our model. 
            file_path = os.path.join(self.directory, folder_name, f"ResNet_{self.date_time}_Epoch_{epoch}.h5")
            self.model.save(file_path)

# Function to train the ResNet. 
# Function input arg 1: num_epochs [int] --> The number of epochs to train the model for. 
# Function input arg 2: batch_size [int] --> The batch_size. 
# Function input arg 3: new_model [bool] --> When True, trains a new model. When False, trains a previous model which you select.
# Function input arg 4: display_plot [bool] --> When True, prints the plots of loss and accuracy. 
def train_ResNet(num_epochs, 
                 batch_size,
                 new_model = True, 
                 display_plot=True):
    
    ##### (1) First, establish paramaters which will be useful for the rest of the code. 
    
    # Load in the date and time. 
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    
    #### (2) Get the user to select the directory in question. 
    
    # Select the training directory.
    directory = select_folder('Please select the training directory') 
    
    # Generate list of images names and corresponding distributions. 
    image_paths, distributions = get_names_and_distributions(directory)
    
    # Split the data into training and testing/validation data. 
    x_train, x_test, y_train, y_test = train_test_split(image_paths,distributions, test_size=0.2)
    
    #### (3) Create or load in a model, as the user desires it. 
    
    # Create the new model.
    if new_model == True:
        
        model = create_resnet(256,256,1)
        
        # Compile the model. 
        KL_loss = tf.keras.losses.KLDivergence()
        optim = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optim,
                      loss = KL_loss, 
                      metrics = ([custom_accuracy(0.1),
                                  custom_accuracy(0.2),
                                  custom_accuracy(0.3)]))
                      
    elif new_model == False: 
        
        # Load in the previously trained model.
        previous_model_path = select_file()
        KL_loss = tf.keras.losses.KLDivergence()
        accuracy_01 = custom_accuracy(0.1)
        accuracy_02 = custom_accuracy(0.2)
        accuracy_03 = custom_accuracy(0.3)
        model = load_model(previous_model_path, custom_objects={'KL_loss' : KL_loss,
                                                                'acc_01' : accuracy_01,
                                                                'acc_02' : accuracy_02,
                                                                'acc_03' : accuracy_03})
        
        # Load in the corresponding pandas data frame containing loss and accuracy.
        # NB: When models get loaded in, their old history is not maintained.
        _, model_name = os.path.split(previous_model_path)
        date_time = re.search('[0-9]+[_][0-9]+', model_name)
        date_time = date_time.group(1)
        df_name = f'training_log_{date_time}.csv'
        df_path = os.path.join(_, df_name)
        df = pd.read_csv(log_data_path, index_col=False)
        
    #### (4) Train the model. 

    # Create our generators. 
    list_idxs = np.arange(len(x_train))
    training_generator = Custom_Generator(x_train, 
                                          y_train, 
                                          batch_size=batch_size,
                                          list_idxs=list_idxs, 
                                          num_channels=1)
    list_idxs = np.arange(len(x_test))
    validation_generator = Custom_Generator(x_test, 
                                            y_test, 
                                            batch_size=batch_size,
                                            list_idxs=list_idxs, 
                                            num_channels=1)

    # Train the model. 
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1, 
                        callbacks=[SaveModelRegularly(interval = 5,
                                                      directory = directory, 
                                                      date_time = date_time)])
    
    #### (5) Log the training data.
    
    # Add the loss and accuracy to the pandas array. 
    df_temporary = pd.DataFrame()
    df_temporary['loss'] = history.history['loss']
    df_temporary['val_loss'] = history.history['val_loss']
    df_temporary['acc_01'] = history.history['acc_01']
    df_temporary['val_acc_01'] = history.history['val_acc_01']
    df_temporary['acc_02'] = history.history['acc_02']
    df_temporary['val_acc_02'] = history.history['val_acc_02']
    df_temporary['acc_03'] = history.history['acc_03']
    df_temporary['val_acc_03'] = history.history['val_acc_03']
    
    # If we need to append our data to that of a previous trainig run, we do it here.
    if new_model == True:
        df = df_temporary
    elif new_model == False:
        df = [df, df_temporary]
        df = pd.concat(df)
        
    # Save the model. 
    folder_name = f'training-data-{date_time}'
    if not os.path.exists(os.path.join(directory, folder_name)):
        os.makedirs(os.path.join(directory, folder_name))
    file_path = os.path.join(directory, folder_name, f"ResNet_{date_time}.h5")
    model.save(file_path) 

    # Save the model history (the loss, accuracy, val_loss and val_accuracy).
    file_path = os.path.join(directory, folder_name, f"training_log_{date_time}.csv")
    df.to_csv(file_path, index=False)
    
    #### (6) Plot the data and save it. 
    
    loss_graph(df['loss'], 
               df['val_loss'], 
               display_plot=display_plot,
               directory=directory, 
               date_time=date_time)
    
    create_accuracy_graph(df['acc_01'], 
                          df['val_acc_01'], 
                          df['acc_02'], 
                          df['val_acc_02'], 
                          df['acc_03'], 
                          df['val_acc_03'], 
                          date_time,
                          display_plot,
                          directory)
    
    #### (7) Let the user know the model has finished training. 
    
    saved_dir = os.path.join(directory, folder_name)
    print(f'██████████████████████\nTraining complete.\n\nModel outputs are stored here:\n\n{saved_dir}_{date_time}\n██████████████████████')

# Function to use the trained ResNet on a folder of images. 
# Function input arg 1: image_ext [list] --> Strings of the image extensions to be considered for processing. 
def use_resnet(image_ext = ['.tif', '.png']):
    
    #### (1) First, we establish variables which will be useful later on. 
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    #### (2) Load in our model. 
    model_path = select_file('Please select the traing model .h5 file')
    KL_loss = tf.keras.losses.KLDivergence()
    acc_01 = custom_accuracy(0.1)
    acc_02 = custom_accuracy(0.2)
    acc_03 = custom_accuracy(0.3)
    model = load_model(model_path, custom_objects={'KL_loss':KL_loss,
                                                   'acc_01':acc_01,
                                                   'acc_02':acc_02,
                                                   'acc_03':acc_03})
    
    #### (3) Get a lit of the image paths we need to consider. 
    directory = select_folder('Please select the folder of images you wish to process')
    image_names = [_ for _ in os.listdir(directory) if any(substring in _ for substring in image_ext)]
    
    #### (4) Iteratively get distributions for each of the images. 
    for i in trange(len(image_names)):
        
        # Load in the image. 
        img = cv2.imread(os.path.join(directory, image_names[i]), -1)
        
        # Scale the image, just as for training. 
        img = (img - img.min()) / (img.max() - img.min()) 
        
        # Add batch and channel dimensions. 
        img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    
        # Save memory and convert to float16. 
        img = img.astype('float16')
        
        # Use our model to get a predicted distribution. 
        distribution = model.predict(img)
        distribution = np.reshape(distribution, (5,))
        
        # Plot and save the distribution 
        image_name = os.path.splitext(image_names[i])[0]
        make_distribution_graph(distribution,
                                directory,
                                display_plot=False,
                                date_time=date_time,
                                name_tag=image_name)

    #### (5) Let the user know the model has finished training. 
    
    folder_name = f'prediction-data-{date_time}'
    saved_dir = os.path.join(directory, folder_name)
    print(f'██████████████████████\nPlotting complete.\n\nModel outputs are stored here:\n\n{saved_dir}\n██████████████████████')