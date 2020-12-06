# Import the necessary packages and modules. 
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import torch
import torch.nn as nn
import math
import numpy as np 

# Test the function 'neural_netowrk()' against the provided outputs. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_neural_network(): 

    ##### (1) Load and prepare data. 
    data =  datasets.load_breast_cancer()
    x, y = data.data, data.target
    
    # Get data dimensions. 
    _, num_features = x.shape
    
    # Split the data into training data and testing data.
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.33, random_state=1234) # Use random_state=1234 arg to generate same data for testing.

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
        
        # Backward pass. 
        # Zero out the gradients. Resetting the gradient is important as by default, PyTorch cumulatively 
        # increases gradients with each backward pass. This is a feature wihch is useful for RNNs, but not 
        # for our model. 
        optimizer.zero_grad()
        
        # Calculate d loss/d x. This is the graident calculation per weight. 
        loss.backward()
        
        # Update the weights.
        optimizer.step()

    ##### (5) Calculate the accuracy of the model. 
    with torch.no_grad():
        y_testing_predicted = model(x_testing)
        y_testing_predicted_classes = y_testing_predicted.round()
        validation_accuracy = y_testing_predicted_classes.eq(y_testing).sum().detach().numpy() / float(y_testing.shape[0])
        
    ##### (6) Test 1: See if the validation accuracy is as expected. 
    assert vaidaltion_accuracy > 0.95, "Test 1 failed. The calculated value of validation accuracy for the model (using a fixed data set) was less than 0.95"

    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_neural_network()
