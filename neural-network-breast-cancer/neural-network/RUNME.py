# Import the necessary packages.
from neural_netowrk_functions import *

# A function which trains itself to predict whether a cancer is metastatic or not by using a neural netowrk. 
# Function inputs arg 1: save_plot --> True or False. When True, saves plot to the img folder of the package. 
# Function inputs arg 2: display_plot --> True or False. When True, displays plot within conole. 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Predicted values --> Outputs a list of predicted values.  
# Function output 3: True values --> Outputs a list of True values corresponding to the predicted values.
neural_network(save_plot=True, 
               display_plot=True): 