import matplotlib.pyplot as plt
import os
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np

# Function inputs arg 1: num_epochs --> The number of iterations over which the model is refined. 
# Function inputs arg 2: loss_array --> Array of size 1 x num_epochs. This array contains the calculated vales of BCE loss made when refining the model with SGD. 
# Function inputs arg 3: save_plot --> True or Flase. When true, saves plot to data directory.  
# Function inputs arg 4: display_plot --> True or Flase. When true, displays the plot. 
# Function output: Graph with the BCE loss per epoch.
def loss_graph(num_epochs, loss_array, save_plot, display_plot):
    
    # Plot the BCE calculated loss per epoch. 
    y = list(range(0,num_epochs))
    plt.plot(y, loss_array)
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('BCE calculated loss', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('Epoch', labelpad=10)

    # Save the plot if the user desires it.
    if save_plot:
        current_directory = os.getcwd()
        file_path = current_directory.replace('logistic-classification-package', 'img')
        file_path = os.path.join(file_path, 'BCE_calculated_loss.png')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (display_plot == False):
        plt.close()
    else:
        plt.show()   

# Function inputs arg 1: display_plot --> Set to True or False. When True, prints graph of BCE calculated loss per epoch.
# Function inputs arg 2: save_plot --> Set to True or False. When True, 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Accuracy --> Outputs the accuracy of the model as a fraction between 0 (innaccurate) and 1 (accurate). 
# Function output 3: Predicted values --> Outputs a list of predicted values.  
# Function output 4: True values --> Outputs a list of True values corresponding to the predicted values.
def logistic_classification(display_plot, save_plot): 

    ##### (1) Load and prepare data. 
    data =  datasets.load_breast_cancer()
    x, y = data.data, data.target
    
    # Get data dimensions. 
    _, num_features = x.shape
    
    # Split the data into training data and testing data.
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.33, random_state=1234)

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
    class LogisticRegression(nn.Module):
        
        def __init__(self, n_input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)
            
        def forward(self, x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted
    
    # Create an instance of our model. 
    model = LogisticRegression(num_features)
    
    ##### (3) Establish the loss and the optimiser. 
    calc_loss = nn.BCELoss() # Use built in binary cross entropy loss function from PyTorch.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # We're using stochastic gradient descent. 
    
    ##### (4) Training loop. 
    num_epochs = 10000
    loss_array = []
    for epoch in range(num_epochs):
    
        # Forward pass: compute the output of the layers given the input data
        y_predicted = model(x_training)
        loss = calc_loss(y_predicted, y_training)
        
        # Log the loss per epoch.
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
    with torch.no_grad():
        y_predicted = model(x_testing)
        y_predicted_classes = y_predicted.round()
        accuracy = y_predicted_classes.eq(y_testing).sum().detach().numpy() / float(y_testing.shape[0])
    
    ##### (6) Plot data associated with the model. 
    loss_graph(num_epochs, loss_array, save_plot, display_plot)
    
    ##### (7) Return data. 
    y_predicted_classes = y_predicted_classes.detach().numpy()
    y_testing = y_testing.detach().numpy()
    
    return model, accuracy, y_predicted_classes, y_testing
    