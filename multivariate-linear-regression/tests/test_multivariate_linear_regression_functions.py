# Import the necessary packages and modules. 
import pandas as pd
from pandas import read_csv 
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import os
from math import sqrt

# Test the function 'multivairate_linear_regression()' against the provided outputs. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_multivariate_linear_regression(): 

    # Reconstruct the path so that we can load in the provided .csv data from the 'data' folder. 
    current_directory = os.getcwd()
    file_path = current_directory.replace('tests', 'data/MV_linear_regression_data.csv')

    # (1) Load in the csv data. 
    data = read_csv(file_path) 
    data = np.array(data)
    
    # Test 1: Test to see if the data has loaded in correctly.
    if data is None: 
        raise TypeError("Test 1 failed. Error opening .csv.'linear_regression_data.csv' is of type None") 
    
        # Split data into X and Y data. 
    _, num_cols = data.shape
    
    # Testing data.
    X_testing = data[:, list(range(1,num_cols))]
    X_testing = np.array(X_testing, dtype=np.float32)
    X_testing = MinMaxScaler().fit_transform(X_testing)
    
    Y_testing = data[:, 0]
    Y_testing = np.array(Y_testing, dtype=np.float32)
    
    # Convert our numpy arrays to tensors. Pytorch requires the use of tensors. 
    rows, cols = X_testing.shape
    x_tensor = torch.from_numpy(X_testing.reshape(rows,cols))
    y_tensor = torch.from_numpy(Y_testing.reshape(rows,1))
    
    ##### (2) Define our model. 
    class MVLinearRegression(torch.nn.Module):
        
        def __init__(self, in_features, out_features):
            super().__init__() # We use 'super()' in the constructor pass in parameters from the parent class.
            self.linear = nn.Linear(in_features, out_features) # Create an object of type linear. 
        
        # The forward function allows us to create a prediction.
        def forward(self, x): 
            return self.linear(x)
    
    # Create an instance of our model. 
    MVLR_model = MVLinearRegression(in_features=cols, out_features=1)
    
    MVLR_model.train()
    
    ##### (3) Establish the loss and the optimiser. 
    calc_MSE = nn.MSELoss() # Use built in loss function from PyTorch.
    learning_rate = 0.001
    optimizer = torch.optim.SGD(MVLR_model.parameters(), lr=learning_rate) # We're using stochastic gradient descent. 
    
    ##### (4) Training loop. 
    num_epochs = 10000
    loss_array = []
    for epoch in range(num_epochs):
    
        # Forward pass: compute the output of the network given the input data
        y_predicted = MVLR_model(x_tensor)
        loss = calc_MSE(y_predicted, y_tensor)
        
        loss_value = loss.detach().numpy()
        loss_value = loss_value.item()
        loss_array.append(loss_value)
        
        # Backward pass: compute the output error with respect to the expected output and then go backward into the network and update the weights using gradient descent.
        loss.backward()
        
        # Update the weights.
        optimizer.step()

        # Zero out the gradients. 
        optimizer.zero_grad()
    
    
    MVLR_model.eval()
    
    
    ##### (5) Test the model. 
    rows, cols = X_testing.shape
    
    # Convert the numpy arrays to tensors. 
    x_test_tensor = torch.from_numpy(X_testing.reshape(rows,cols))
    y_test_tensor = torch.from_numpy(Y_testing.reshape(rows ,1))
    y_pred_tensor = MVLR_model(x_test_tensor)
    RMSE = sqrt(calc_MSE(y_pred_tensor, y_test_tensor).detach().numpy())

    y_pred = y_pred_tensor.detach().numpy()
    y_test = y_test_tensor.detach().numpy()
    R2 = r2_score(y_test, y_pred)
    
    ##### (7) Extract the y_intercept and coefficients calculated by our multuvariate linear regression model.
    coefficients, y_intercept = MVLR_model.parameters()
    coefficients = coefficients.data.detach().numpy()
    y_intercept = y_intercept.data.detach().numpy()
    output = np.append(y_intercept, coefficients)
    
    # Test 2: See if the values of  RMSE, R2 and the weigths for the linear model fall within the expeted range. 
    assert 2.2 < RMSE < 2.6, "Test 2.1 failed. The calculated values of RMSE fell outside of the following expected range: 2.2 < RMSE < 2.6."
    assert 0.9 < R2 <= 1, "Test 2.2 failed. The calculated values of R2 fell outside of the following exected range: 0.9 < R2 <= 1."
    assert 15 < output[0] < 17.5, "Test 2.3 failed. The bias (the y-intercept) of the linear model fell outside of the following expected range: 15 < bias < 17.5."
    assert -17 < output[1] < -15, "Test 2.4 failed. The weight correspoding to the first training column fell outside of the following expected range: -17 < weight1 < -15."
    assert 23 < output[2] < 25.5, "Test 2.5 failed. The weight correspoding to the second training column fell outside of the following expected range: 23 < weight2 < 25.5."

    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_multivariate_linear_regression()