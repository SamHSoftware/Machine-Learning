import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import torch
import torch.nn as nn
import math

# Function inputs arg 1: num_epochs --> The number of iterations over which the model is refined. 
# Function inputs arg 2: training_loss --> Array of size 1 x num_epochs. This array contains the calculated values of BCE loss made when refining the model with SGD. 
# Function inputs arg 3: validation_loss --> Array of size 1 x num_epochs. This array contains the calculated values of BCE loss calculated for validation. 
# Function inputs arg 4: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 5: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Graph with the BCE loss per epoch.
def loss_graph(num_epochs, 
               training_loss, 
               validation_loss, 
               save_plot, 
               display_plot):
    
    # Plot the BCE calculated loss per epoch. 
    y = list(range(0,num_epochs))
    plt.plot(y, training_loss, label="Training loss")
    plt.plot(y, validation_loss, label="Validation loss")
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('BCE calculated loss', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot if the user desires it.
    if save_plot:
        current_directory = os.getcwd()
        file_path, _ = os.path.split(current_directory)
        file_path = os.path.join(file_path, 'img', 'training_and_validation_loss.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   
        
# Function inputs arg 1: num_epochs --> The number of iterations over which the model is refined. 
# Function inputs arg 2: training_accuracy --> Array of size 1 x num_epochs. This array contains the calculated values of training accuracy. 
# Function inputs arg 3: validation_accuracy --> Array of size 1 x num_epochs. This array contains the calculated values of validation accuracy. 
# Function inputs arg 4: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 5: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Graph with the training and validation accuracy per epoch.
def accuracy_graph(num_epochs, 
               training_accuracy, 
               validation_accuracy, 
               save_plot, 
               display_plot):
    
    # Plot the BCE calculated loss per epoch. 
    y = list(range(0,num_epochs))
    plt.plot(y, training_accuracy, label="Training accuracy")
    plt.plot(y, validation_accuracy, label="Validation accuracy")
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Accuracy', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot if the user desires it.
    if save_plot:
        current_directory = os.getcwd()
        file_path, _ = os.path.split(current_directory)
        file_path = os.path.join(file_path, 'img', 'training_and_validation_accuracy.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   
        
# This function creates a confusion matrix to help assess the model. 
# Function inputs arg 1: cm --> The confusion matrix as generated by the function 'confusion_matrix()'
# Function inputs arg 2: classes --> Tuple of strings to label class identities on the plot.  
# Function inputs arg 3: normalize --> True or Flase. When true, data is normalized between 0 and 1 relative to the total of each row.
# Function inputs arg 4: title --> A string. 
# Function inputs arg 5: cmap --> The chosen colormap. 
# Function inputs arg 6: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 7: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Figure with the confusion matrix. 
def plot_confusion_matrix(cm, 
                          classes,
                          save_plot=True,
                          display_plot=True,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    # Save the plot if the user desires it.
    if save_plot:
        current_directory = os.getcwd()
        file_path, _ = os.path.split(current_directory)
        file_path = os.path.join(file_path, 'img', 'confusion_matrix.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# A function which trains itself to predict whether a cancer is metastatic or not by using a neural netowrk. 
# Function inputs arg 1: save_plot --> True or False. When True, saves plot to the img folder of the package. 
# Function inputs arg 2: display_plot --> True or False. When True, displays plot within conole. 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Predicted values --> Outputs a list of predicted values.  
# Function output 3: True values --> Outputs a list of True values corresponding to the predicted values.
def neural_network(save_plot=True, 
                   display_plot=True): 
    
    ##### (1) Load and prepare data. 
    data =  datasets.load_breast_cancer()
    x, y = data.data, data.target
    
    # Get data dimensions. 
    _, num_features = x.shape
    
    # Split the data into training data and testing data.
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.33) # Use random_state=1234 arg to generate same data for testing.

    # Scale the data. 
    x_training = StandardScaler().fit_transform(x_training)
    x_testing = StandardScaler().fit_transform(x_testing)
    
    # Convert data to tensors.
    x_training = torch.from_numpy(x_training.astype(np.float32))
    x_testing = torch.from_numpy(x_testing.astype(np.float32))
    
    num_samples = y_training.shape
    y_training = torch.from_numpy(y_training.reshape(num_samples[0], 1))
    y_training = y_training.type(torch.float32)
                                 
    num_samples = y_testing.shape
    y_testing = torch.from_numpy(y_testing.reshape(num_samples[0], 1))
    y_testing = y_testing.type(torch.float32)     

    ##### (2) Create our model. 
    class NeuralNetwork(nn.Module): 
        def __init__(self, num_features): 
            super(NeuralNetwork, self).__init__()
            self.linear_1 = nn.Linear(num_features, math.floor(num_features/2))
            self.linear_2 = nn.Linear(math.floor(num_features/2), math.floor(num_features/4))
            self.linear_3 = nn.Linear(math.floor(num_features/4), 1)

            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            output_1 = self.sigmoid(self.linear_1(x))
            output_2 = self.sigmoid(self.linear_2(output_1))
            y_predicted = self.sigmoid(self.linear_3(output_2))
            return y_predicted 
        
    # Create an instance of our model. 
    model = NeuralNetwork(num_features)
    
    ##### (3) Establish the loss and the optimiser. 
    calc_loss = nn.BCELoss() # Use built in binary cross entropy loss function from PyTorch.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # We're using stochastic gradient descent. 
   
    ##### (4) Training loop. 
    num_epochs = 10000
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    for epoch in range(num_epochs):
    
        # Forward pass: compute the output of the layers given the input data
        y_training_predicted = model(x_training)
        
        # Log the training loss per epoch.
        loss = calc_loss(y_training_predicted, y_training)
        loss_value = loss.detach().numpy()
        loss_value = loss_value.item()
        training_loss.append(loss_value)
        
        # Log the validation loss per epoch. 
        y_testing_predicted = model(x_testing)
        loss = calc_loss(y_testing_predicted, y_testing)
        loss_value = loss.detach().numpy()
        loss_value = loss_value.item()
        validation_loss.append(loss_value)
        
        # Log the training accuracy per epoch. 
        y_training_predicted_classes = y_training_predicted.round()
        accuracy = y_training_predicted_classes.eq(y_training).sum().detach().numpy() / float(y_training.shape[0])
        training_accuracy.append(accuracy)
        
        # Log the validation accuracy per epoch. 
        y_testing_predicted_classes = y_testing_predicted.round()
        accuracy = y_testing_predicted_classes.eq(y_testing).sum().detach().numpy() / float(y_testing.shape[0])
        validation_accuracy.append(accuracy)

        # Backward pass. 
        # Zero out the gradients. Resetting the gradient is important as by default, PyTorch cumulatively 
        # increases gradients with each backward pass. This is a feature wihch is useful for RNNs, but not 
        # for our model. 
        optimizer.zero_grad()
        
        # Calculate d loss/d x. This is the graident calculation per weight. 
        loss.backward()
        
        # Update the weights.
        optimizer.step()

    ##### (5) Plot data associated with the model. 
    
    # Plot the loss graph. 
    loss_graph(num_epochs, 
               training_loss, 
               validation_loss, 
               save_plot, 
               display_plot)
    
    # Plot the accuracy graph. 
    accuracy_graph(num_epochs, 
               training_accuracy, 
               validation_accuracy, 
               save_plot, 
               display_plot)
    
    # Plot the confusion matrix.
    confusion = confusion_matrix(y_testing.detach().numpy(), y_predicted_classes.detach().numpy())
    names = ('Malignant', 'Benign')
    plt.figure()
    plot_confusion_matrix(confusion, 
                          names, 
                          save_plot, 
                          display_plot)
    
    ##### (6) Return data. 
    y_predicted_classes = y_predicted_classes.detach().numpy()
    y_testing = y_testing.detach().numpy()
    
    return model, y_predicted_classes, y_testing
    