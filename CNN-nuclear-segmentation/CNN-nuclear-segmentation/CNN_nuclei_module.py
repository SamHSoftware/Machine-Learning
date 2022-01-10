from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import imageio
from PIL import Image 
from PIL import ImageFont
from PIL import ImageDraw
import cv2 
from tqdm import trange 
import numpy as np
from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
import tensorflow as tf 
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import skimage
from skimage import exposure
import keras 
from keras.utils.np_utils import to_categorical
from keras.models import load_model 
from datetime import datetime 
import pandas as pd 

# A function to allow the user to select the folder contianing the data.
# Function inputs args: None. 
# Function output 1: The path of that the folder selected by the user. 
def folder_selection_dialog():
    root = Tk()
    root.title('Please select the directory containing the images')
    root.filename = filedialog.askdirectory(initialdir="/", title="Select A Folder")
    directory = root.filename
    root.destroy()

    return directory

# A function to allow the user to select the model they wish to use or retrain. 
# Function inputs args: None. 
# Function output 1: The file path of that which was selected by the user. 
def file_selection_dialog():
    root = Tk()
    root.title('Please select the machine learning model in question')
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=[("All files", "*.*")])
    file_path = root.filename
    root.destroy()

    return file_path

# Function to display and save the training loss and validation loss per epoch.
# Function input arg 1: training_loss --> Array of size 1 x num_epochs. This array contains the calculated values of loss for training. 
# Function input arg 2: validation_loss --> Array of size 1 x num_epochs. This array contains the calculated values of loss for validation. 
# Function input arg 3: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function input arg 4: display_plot --> True or Flase. When true, displays the plot. 
# Function input arg 5: directory --> The directory containing the training dataset. 
# Function input arg 6: date_time --> The datetime string in the format of 'YMD_HMS'. 
def loss_graph(training_loss, 
               validation_loss, 
               save_plot, 
               display_plot,
               directory, 
               date_time):
    
    # Plot the loss per epoch. 
    y = list(range(0,len(training_loss)))
    plt.plot(y, training_loss, label = "Training loss")
    plt.plot(y, validation_loss, label = "Validation loss")
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Loss', labelpad=10) # The labelpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot if the user desires it.
    if save_plot:
        folder_name = f'training data_{date_time}'
        if not os.path.exists(os.path.join(directory, folder_name)):
            os.makedirs(os.path.join(directory, folder_name))
        file_path = os.path.join(directory, folder_name, f'loss_{date_time}.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()

# Function to make a movie from the training data, showing the model training over time.
# Function input arg 1: directory --> The directory containing the training dataset. 
# Function input arg 2: date_time --> The datetime string in the format of 'YMD_HMS'. 
# Function input arg 3: make_the_movie --> If True, a movie will be made and saved to directory_{date_time}. 
def make_movie(directory, 
               date_time, 
               make_the_movie=True):
    
    if make_the_movie: 

        # Determine which images we'll need to load in. 
        folder_name = f'training data_{date_time}'
        image_names = [_ for _ in os.listdir(os.path.join(directory, folder_name)) if 'predicted.png' in _]
        number_of_frames = len(image_names)

        # Sort out a couple of paths and fonts. 
        gif_full_path = os.path.join(directory, folder_name, 'animation.gif')
        font = ImageFont.truetype("arial.ttf", 100)

        # Create the raw image. 
        raw_img = Image.open(os.path.join(directory, folder_name, 'raw_image.png'))
        draw = ImageDraw.Draw(raw_img)
        draw.text((20, 20),f'Raw image',font=font, fill=(255))
        array = np.array(raw_img)
        cmap = plt.cm.gray
        raw_image = (cmap(array)*255).astype(np.uint8)

        # Create the ground truth labelled image. 
        gtruth_img = Image.open(os.path.join(directory, folder_name, 'gtruth_image.png'))
        draw = ImageDraw.Draw(gtruth_img)
        draw.text((20, 20),f'Ground truth image',font=font, fill=(180))
        array = np.array(gtruth_img)
        cmap = plt.cm.gnuplot2
        gtruth_image = (cmap(array)*255).astype(np.uint8)

        with imageio.get_writer(gif_full_path, mode='I') as writer:
            for x in trange(number_of_frames):

                # Create the predicted image. 
                predicted_img = Image.open(os.path.join(directory, folder_name, f'{str(x)}_predicted.png'))
                draw = ImageDraw.Draw(predicted_img)
                draw.text((20, 20),f'Predicted image',font=font, fill=(180))
                draw.text((20, 2030),f'Epoch: {str(x)}',font=font, fill=(180))
                array = np.array(predicted_img)
                cmap = plt.cm.gnuplot2
                predicted_image = (cmap(array)*255).astype(np.uint8)

                # Join all the images together. 
                image = cv2.hconcat([raw_image, gtruth_image, predicted_image])
                
                # Save the image. 
                writer.append_data(image)

# Function to display and save the training accuracy and validation accuracy per epoch.
# Function input arg 1: training_accuracy --> Array of size 1 x num_epochs. This array contains the calculated values of training accuracy. 
# Function input arg 2: validation_accuracy --> Array of size 1 x num_epochs. This array contains the calculated values of validation accuracy. 
# Function input arg 3: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function input arg 4: display_plot --> True or Flase. When true, displays the plot. 
# Function input arg 5: directory --> The directory containing the training dataset. 
# Function input arg 6: date_time --> The datetime string in the format of 'YMD_HMS'. 
def accuracy_graph(training_accuracy, 
                   validation_accuracy, 
                   save_plot, 
                   display_plot,
                   directory, 
                   date_time):
    
    # Plot the BCE calculated loss per epoch. 
    y = list(range(0,len(training_accuracy)))
    plt.plot(y, training_accuracy, label="Training accuracy")
    plt.plot(y, validation_accuracy, label="Validation accuracy")
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Accuracy', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot if the user desires it.
    if save_plot:
        folder_name = f'training data_{date_time}'
        if not os.path.exists(os.path.join(directory, folder_name)):
            os.makedirs(os.path.join(directory, folder_name))
        file_path = os.path.join(directory, folder_name, f'accuracy_{date_time}.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# A function which will append images within a directory into a numpy array. These imags will also be standardized. 
# Function input 1: image_list [list of strings] --> Each item in the list is the name of an image which needs to be appended into one stack e.g. image1.tif.
# Function input 2: directory [string] --> The directory containing the images.
# Function input 3: raw [bool] --> When true, will standardize the image mean to 0, and set standard deviation to 1. 
# Function output 1: image stack [numpy array] --> The 3D stack of appended images. 
# Function output 2: num_classes_out [numpy array] --> The unique values of the images. Used for determining the number of classes.
def append_images(image_list,
                  directory, 
                  raw=True):

    # Create an empty list. 
    image_stack = []
    num_classes_out = 0
    
    # Iterate through the images of our list and append them to our stack. 
    for i in range(len(image_list)):
        file_path = os.path.join(directory, image_list[i])
        img = cv2.imread(file_path, -1)
        
        # For raw images. 
        if raw: 
            
            # Scale the image between 0 and 1. 
            img = (img - img.min()) / (img.max() - img.min()) 
            
            # Höfener, H., Homeyer, A., Weiss, N., Molin, J., Lundström, C.F. and Hahn, H.K., 2018. Deep learning nuclei detection: A simple approach can deliver state-of-the-art results. Computerized Medical Imaging and Graphics, 70, pp.43-52.
            # Mirror the data horizontally...
            for h in range(2):
                if h == 0: 
                    img2 = img
                else:
                    img2 = np.flip(img, axis=0) 

                # ... and vertically. 
                for v in range(2):
                    if v == 0: 
                        img3 = img2
                    else: 
                        img3 = np.flip(img2, axis=1)
            
                    # Add an extra axis (for processing later) and append our image to the stack.
                    img3 = np.stack((img3,)*1, axis=-1)
                    image_stack.append(img3)

        # For gtruth images. 
        else: 
            num_classes_out = len(np.unique(img))
            
            # Mirror the data horizontally...
            for h in range(2):
                if h == 0: 
                    img2 = img
                else:
                    img2 = np.flip(img, axis=0) 

                # ... and vertically. 
                for v in range(2):
                    if v == 0: 
                        img3 = img2
                    else: 
                        img3 = np.flip(img2, axis=1)
            
                    # Aappend our image to the stack.
                    image_stack.append(img3)
    
    # Convert the stack to a numpy array. 
    image_stack = np.asarray(image_stack)

    return image_stack, num_classes_out 

# Function to create our Unet model. 
# Function input 1: n_classes [int] --> Number of classes which need to be classified. 
# Function input 2: img_height [int] --> Image height in pixels. 
# Function input 3: img_width [int] --> Image width in pixels. 
# Function input 4: img_channels [int] --> Number of channels. For a grayscale image, this would be 1. for an RGB image, this would be 3.
# Function output 1: The untrained model. 
def multiclass_Unet(n_classes,
                   img_height,
                   img_width,
                   img_channels):

    inputs = Input((img_height, img_width, img_channels))
    #print("inputs:", inputs.shape)
    
    # Contraction path. 
    c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c1)
    #print("p1:", p1.shape)
    
    c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D(2,2)(c2)
    #print("p2:", p2.shape)

    c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c3)
    #print("p3:", p3.shape)

    c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c4)
    #print("p4:", p4.shape)

    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (2,2), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    #print("c5:", c5.shape)

    # Expansion path. 
    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    #print("u6:", u6.shape)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    #print("c6:", c6.shape)

    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    #print("u7:", u7.shape)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    #print("c7:", c7.shape)
    
    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    #print("u8:", u8.shape)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    #print("c8:", c8.shape)

    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    #print("u9:", u9.shape)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    #print("c9:", c9.shape)

    outputs = Conv2D(n_classes, (1,1), activation='softmax')(c9)
    outputs = tf.reshape(outputs, [-1, img_height*img_width, n_classes]) # This rehsape is necessary to use sample_weights. 
    #print("outputs:", outputs.shape)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# A function capable of training a CNN to classifying pixels within .tif microscopy images of cell nuclei. 
# Function input 1: directory [str] --> The directory containing the original and gtruth data. 
# Function input 2: save_plot [bool] --> When True, graphical data will be saved. 
# Function input 3: display_plot [bool] --> When True, graphical data will be displayed in the console. 
# Function input 4: save_model [bool] --> When True, saves the model to the directory containing the training data. 
# Function input 5: train_previous_model [bool] --> When True, the user is prompted to select a previously trained model, in order to continue it's training.
# Function input 6: num_epochs [int] --> The number of epochs to train the model.
# Function input 7: make_movie [bool] --> When true, outputs a movie of the model learning over time. 
def train_CNN(directory,
              save_plot=True,
              display_plot=True,
              save_model=True, 
              train_previous_model=False,
              num_epochs=500, 
              make_the_movie=True):
    
    #### (1) Establish variables important for the code. 
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    
    #### (2) Create our training and testing dataset. 
    
    # Get the names of our raw and labelled (gtruth) images. 
    raw_images = [image for image in os.listdir(directory) if all([image.endswith('.tif'), 'gtruth' not in image])]    
    gtruth_images = [image for image in os.listdir(directory) if all([image.endswith('.tif'), 'gtruth' in image])]
    
    # Get the images (X) and their ground truth equivalents (Y).
    Y, number_classes = append_images(gtruth_images, directory, raw = False)
    X, _ = append_images(raw_images, directory, raw = True)
    
    # Encode our labels, to ensure that the that the first label value starts from 0 (not 1) as the model expects.
    label_encoder = LabelEncoder()
    slices, height, width = Y.shape
    Y_reshaped = Y.ravel() # Reshape each image into a single column. 
    Y_reshaped_encoded = label_encoder.fit_transform(Y_reshaped)
    Y_reshaped_encoded2 = Y_reshaped_encoded.reshape(slices, height, width)
    
    # Add an additional dimension to our ground truth data, as the model expects it. 
    Y = np.expand_dims(Y_reshaped_encoded2, axis = 3)
    
    # Convert our ground truth pixel values to a one-hot-encoded format. For instance, a pixel value of 2 would be converted to [0,0,1,0]. This is needed for loss functions such as categorical cross entropy loss functions.
    Y_categorical = to_categorical(Y, number_classes)
    Y_categorical = Y_categorical.reshape((Y.shape[0], Y.shape[1], Y.shape[2], number_classes))
    Y_categorical = np.reshape(Y_categorical, (Y_categorical.shape[0], Y_categorical.shape[1]*Y_categorical.shape[2], Y_categorical.shape[3]))
    
    # Split our data into test and train datasets. 
    x_train, x_test, y_train, y_test = train_test_split(X,Y_categorical, test_size=0.5)
    
    #### (3) Define our new model, or load in a previous model to continue it's training. 
    
    # If we want to make a new model...
    if train_previous_model == False: 
        
        # Create the model.
        img_height = x_train.shape[1]    
        img_width = x_train.shape[2]
        img_channels = x_train.shape[3]
        model = multiclass_Unet(n_classes = number_classes,
                                img_height = img_height,
                                img_width = img_width,
                                img_channels = img_channels)

        focal_loss = sm.losses.CategoricalFocalLoss()
        model.compile(optimizer='adam', 
                      loss=focal_loss, 
                      metrics=['accuracy'])
    
    # If we want to continue training a previous model... 
    elif train_previous_model == True:
        
        # Load in the previously trained model.
        previous_model_path = file_selection_dialog()
        focal_loss = sm.losses.CategoricalFocalLoss()
        model = load_model(previous_model_path, custom_objects={'focal_loss': focal_loss})
        
        # Load in the corresponding pandas data frame containing loss and accuracy.
        # NB: When models get loaded in, their old history is not maintained.
        _, model_name = os.path.split(previous_model_path)
        date_time = re.search('[0-9]+[_][0-9]+', model_name)
        date_time = date_time.group(1)
        df_name = f'loss_accuracy_{date_time}.csv'
        df_path = os.path.join(_, df_name)
        df = pd.read_csv(log_data_path, index_col=False)
    
    # Create a callback such that the model will save an image montage per epoch. 
    class SaveMontageCallBack(keras.callbacks.Callback):
        def __init__(self, x_test, y_test, img_height, img_width, directory, date_time):
            self.x_test = np.expand_dims(x_test[0,:,:,:], 0)
            self.y_test = y_test[0,:,:]
            self.img_height = img_height
            self.img_width = img_width
            self.directory = directory
            self.date_time = date_time

        def on_epoch_end(self, epoch, logs={}): 
            
            # Create the directory within which we can save the images. 
            folder_name = f'training data_{self.date_time}'
            if not os.path.exists(os.path.join(directory, folder_name)):
                os.makedirs(os.path.join(directory, folder_name))

            # Save the raw and ground truth images (we just need to do it once).
            if epoch == 0:
                # Save the raw image.
                raw_image = skimage.exposure.equalize_adapthist(np.squeeze(self.x_test), clip_limit=0.03)
                raw_image = (raw_image*255).astype('uint8')
                file_path = os.path.join(directory, folder_name, 'raw_image.png')
                im = Image.fromarray(raw_image)
                im.save(file_path)
                
                # Save the ground truth image. 
                gtruth_image = (np.argmax(np.squeeze(self.y_test), axis=1)).reshape(self.img_height, self.img_width)
                gtruth_image = (gtruth_image / np.amax(gtruth_image)) * 255
                gtruth_image = gtruth_image.astype('uint8')
                file_path = os.path.join(directory, folder_name, 'gtruth_image.png')
                im = Image.fromarray(gtruth_image)
                im.save(file_path)

            # Save the predicted image.
            y_pred = model.predict(self.x_test)
            y_pred_argmax = np.argmax(y_pred, axis=2)
            y_pred_argmax = np.reshape(y_pred_argmax, (self.img_height, self.img_width))
            y_pred_argmax = (y_pred_argmax / np.amax(y_pred_argmax)) * 255
            y_pred_argmax = y_pred_argmax.astype('uint8')
            file_name = f'{str(epoch)}_predicted.png'
            file_path = os.path.join(directory, folder_name, file_name)
            im = Image.fromarray(y_pred_argmax)
            im.save(file_path)
            
    save_montage = SaveMontageCallBack(x_test, y_test, img_height, img_width, directory, date_time)
            
    #### (4)) Train our model.
    
    history = model.fit(x_train,
                        y_train,
                        batch_size=1,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=(x_test,y_test), 
                        callbacks=[save_montage])
    
    # Add the loss and accuracy to the pandas array. 
    df_temporary = pd.DataFrame()
    df_temporary['loss'] = history.history['loss']
    df_temporary['accuracy'] = history.history['accuracy']
    df_temporary['val_loss'] = history.history['val_loss']
    df_temporary['val_accuracy'] = history.history['val_accuracy']
    
    # If we need to append our data to that of a previous trainig run, we do it here.
    if train_previous_model == False:
        df = df_temporary
    elif train_previous_model == True:
        df = [df, df_temporary]
        df = pd.concat(df)
        
    # If the user desires it, save the model as a SavedModel, and save the loss and accuracy values, such that they can be referred to in the instance that a model is loaded in. 
    if save_model == True:
        
        # Save the model. 
        folder_name = f'training data_{date_time}'
        if not os.path.exists(os.path.join(directory, folder_name)):
            os.makedirs(os.path.join(directory, folder_name))
        file_path = os.path.join(directory, folder_name, f"multiclass_CNN_{date_time}.hdf5")
        model.save(file_path) 
        
        # Save the model history (the loss, accuracy, val_loss and val_accuracy).
        file_path = os.path.join(directory, folder_name, f"loss_accuracy_{date_time}.csv")
        df.to_csv(file_path, index=False)
        
    #### (5) Assess our model performance. 
    
    # Create the loss graph. 
    loss_graph(df['loss'], 
               df['val_loss'], 
               save_plot, 
               display_plot,
               directory, 
               date_time)
    
    # Create the accuracy graph. 
    accuracy_graph(df['accuracy'], 
                   df['val_accuracy'], 
                   save_plot, 
                   display_plot,
                   directory, 
                   date_time)
    
    # Create the movie. 
    make_movie(directory, 
               date_time, 
               make_the_movie)
    
    #### (6) State script completion.
    
    print(f'===================\nThe code has finished running.\n\nPlease refer to the following directory for the ouputs:\n\nf"{directory}_{date_time}"\n\nThanks for using the code!\n===================')

# Function to use a trained CNN to classify data and save the results in a new directory.
def use_CNN():
    
    #### (1) Establish variables important for the code. 
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    
    # Load in the previously trained model.
    previous_model_path = file_selection_dialog()
    focal_loss = sm.losses.CategoricalFocalLoss()
    model = load_model(previous_model_path, custom_objects={'focal_loss': focal_loss})
    
    # Get the user to select the directory containing the images which need to be classified. 
    directory = folder_selection_dialog()
    
    # List images in the directory.
    image_names = [_ for _ in os.listdir(directory)]
    
    # Append all the raw images together
    X, _ = append_images(image_names, directory, raw = True)
    
    # Extract image features. 
    img_height = X.shape[1]
    img_width = X.shape[2]
    
    # Iterate through the images, classify their pixels, then save the results.
    for i in trange(X.shape[0]):
        
        # Make predictions for each image. 
        image = np.reshape((X[i,:,:,:]), (1, img_height, img_width, 1))
        y_pred = model.predict(image)
        y_pred_argmax = np.argmax(y_pred, axis=2)
        y_pred_argmax = np.reshape(y_pred_argmax, (img_height, img_width)).astype(np.uint8)
        
        # Save the predicted image.
        file_name = image_names[i]
        file_path = os.path.join(directory, 'classified_images', file_name)
        
        if not os.path.exists(os.path.join(directory, 'classified_images')):
            os.makedirs(os.path.join(directory, 'classified_images'))

        im = Image.fromarray(y_pred_argmax)
        im.save(file_path)
    
    # Print a completion statement. 
    print(f'===================\nThe code has finished running.\n\nPlease refer to the following directory for the ouputs:\n\nf"{directory}_classified_images"\n\nThanks for using the code!\n===================')