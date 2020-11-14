from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import ntpath
from pandas import read_csv 
import torch
from torch import nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from math import sqrt

# A function to allow the user to select the image they wish to analyse. 
# Function inputs args: None. 
# Function output 1: The file path of that which was selected by the user. 
def file_selection_dialog():
    root = Tk()
    root.title('Please select the file in question')
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=[("All files", "*.*")])
    file_path = root.filename
    root.destroy()

    return file_path

# Function inputs arg 1: file_path --> String. File path to the .csv file wihtin which the data is stored. 
# Function inputs arg 2: predictions --> Array of size 1xn. The predicted values outputted by the model. 
# Function inputs arg 3: test --> Array of size 1xn. The test values to compare against those outputted by the model. 
# Function inputs arg 4: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 5: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Graph with the mean square error value per epoch. 
def connectpoints_graph(file_path, predictions, test, save_plot, display_plot):
    
    # Get the number of rows. 
    rows, _ = predictions.shape
    
    # Organise the prediction and truth data. 
    predictions = np.vstack(np.concatenate(predictions, axis=0)) 
    test = np.vstack(test)   
    
    y_values = np.empty((rows*2, 1), dtype=float32)
    y_values[0::2] = test
    y_values[1::2] = predictions
    
    x_values = np.empty((rows*2, 1), dtype=float32)
    x_values[0::2] = 0
    x_values[1::2] = 1

    for i in range(0, len(x_values), 2):
        plt.plot(x_values[i:i+2], y_values[i:i+2], '-ok', mfc='r', mec='k')
    
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Test and prediction values', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xticks([0,1], ['Test \ndata', 'Prediction \ndata'], rotation=0)
    plt.xlim(-0.2, 1.2)
    
    # Save the plot if the user desires it.
    if save_plot:
        _, tail = ntpath.split(file_path)
        new_file_path = file_path.replace('.csv', '_evaluation.png')
        plt.savefig(new_file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show() 
        
# Function inputs arg 1: file_path --> String. File path to the .csv file wihtin which the data is stored. 
# Function inputs arg 2: num_epochs --> The number of iterations over which the model is refined. 
# Function inputs arg 3: loss_array --> Array of size 1 x num_epochs. This array contains the calculated vales of MSE made when refining the model with SGD. 
# Function inputs arg 4: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 5: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Graph with a comparison between the original test values and their counterparts predicted by the model. 
def loss_graph(file_path, num_epochs, loss_array, save_plot, display_plot):
    
    # Plot the MSE calculated loss per epoch. 
    y = list(range(0,num_epochs))
    plt.plot(y, loss_array)
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('MSE calculated loss', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)

    # Save the plot if the user desires it.
    if save_plot:
        _, tail = ntpath.split(file_path)
        new_file_path = file_path.replace('.csv', '_MSE_loss.png')
        plt.savefig(new_file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# Function inputs args 1: file_path --> Input as string. The file path for the data in question.
# Function inputs args 2: display_plot --> Set to True or False. When true, displays the graphs. 
# Function inputs args 3: save_plot --> Set to True or False. When True, saves graphs to file_path folder.
# Function output 1: The trained multivariate linear regression model. The model exects an input tensor of dtype float32.
# Function output 2: The weights and biases for the multivariate linear regression model, ordered as the bias, then w1, w2, ...wn for features x1, x2, ... xn.
# Function output 3: RMSE between test data and truth data. 
# Function output 4: R2 between test data and truth data. 
def MV_linear_regression(file_path, display_plot, save_plot): 

    ##### (1) Load and prepare data. 
    data = read_csv(file_path)
    
    # Scale data, and split the data into training data and testing data.
    data = np.array(data)
    
    np.random.shuffle(data)
    training_data, testing_data = train_test_split(data,test_size=0.33)

    # Split data into X and Y data. 
    _, num_cols = data.shape
    
    # Testing data.
    X_testing = testing_data[:, list(range(1,num_cols))]
    X_testing = np.array(X_testing, dtype=np.float32)
    X_testing = MinMaxScaler().fit_transform(X_testing)
    
    Y_testing = testing_data[:, 0]
    Y_testing = np.array(Y_testing, dtype=np.float32)
    
    # Training data.
    X_training = training_data[:, list(range(1,num_cols))] 
    X_training = np.array(X_training, dtype=np.float32)
    X_training = MinMaxScaler().fit_transform(X_training)
    
    Y_training = training_data[:, 0]
    Y_training = np.array(Y_training, dtype=np.float32)
    
    # Convert our numpy arrays to tensors. Pytorch requires the use of tensors. 
    rows, cols = X_training.shape
    x_tensor = torch.from_numpy(X_training.reshape(rows,cols))
    y_tensor = torch.from_numpy(Y_training.reshape(rows,1))
    
    ##### (2) Define our model. 
    class MVLinearRegression(torch.nn.Module):
        
        def __init__(self, in_features, out_features):
            super().__init__() # We use 'super()' in the constructor pass in parameters from the parent class.
            
            # Here, were defining a linear function which maps data from an in_features-dimensional space to an out_features-dimensional space. 
            # To move between these spaces, we apply a weight matrix to the in_features-dimensional tensor.
            # The weight matrix is out_features - by - out_features in size. 
            # The weight matrix is stored in the PyTorch linear layer class.  
            self.linear = nn.Linear(in_features, out_features) # Within the constructor, we define a layer as a class attribute. 
        
        # Forward propogation: The process of transforming an input tensor to an output tensor (our prediction).
        # The forward function passes the input tensor through our model in the forward direction. This is called forward propogation. 
        # Here, our function accepts a tensor as an input, then returns another tensor as an output.
        def forward(self, x): 
            # I think that by using forward, e run through all the layers specified in the constructor. 
            # So for this module, we pass our tensor into the linear layer. 
            return self.linear(x)
    
    # Create an instance of our model. 
    MVLR_model = MVLinearRegression(in_features=cols, out_features=1)
    
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
            
    ##### (5) Test the model. 
    rows, cols = X_testing.shape
    
    # Calculate the root mean squaure error between predicted values and truth values. 
    x_test_tensor = torch.from_numpy(X_testing.reshape(rows,cols))
    y_test_tensor = torch.from_numpy(Y_testing.reshape(rows ,1))
    y_pred_tensor = MVLR_model(x_test_tensor)
    RMSE = sqrt(calc_MSE(y_pred_tensor, y_test_tensor).detach().numpy()) 
    
    # Calculate the R2 value between predicted values and truth values. 
    y_pred = y_pred_tensor.detach().numpy()
    y_test = y_test_tensor.detach().numpy()
    R2 = r2_score(y_test, y_pred) 
    
    ##### (6) Plot data associated with the model. 
    
    # Display a graph of the mean square error value per epoch. 
    loss_graph(file_path, num_epochs, loss_array, save_plot, display_plot)
    
    # Display a comparison between the original test values and their counterparts predicted by the model. 
    connectpoints_graph(file_path, y_pred, y_test, save_plot, display_plot)
    
    ##### (7) Extract the bias and weights calculated by our multuvariate linear regression model.
    coefficients, y_intercept = MVLR_model.parameters()
    coefficients = coefficients.data.detach().numpy()
    y_intercept = y_intercept.data.detach().numpy()
    output = np.append(y_intercept, coefficients)
    
    return MVLR_model, output, RMSE, R2