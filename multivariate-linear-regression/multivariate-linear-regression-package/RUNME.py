# Import the necessary packages.
from multivariate_linear_regression_functions import *

# A function to allow the user to select the data they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the file in question. 
file_path = file_selection_dialog()

# A function to compare the effects of different image filters.
# Function inputs args 1: file_path --> Input as string. The file path for the data in question.
# Function inputs args 2: plot_images --> Set to True or False. When true, displays the graphs. 
# Function inputs args 3: save_plot --> Set to True or False. When True, saves graphs to file_path folder.
# Function output 1: The trained model multivariate linear regression model.  
# Function output 2: The weights and biases for the multivariate linear regression model, ordered as the bias, then w1, w2, ...wn for features x1, x2, ... xn.
# Function output 3: RMSE between test data and truth data. 
# Function output 4: R2 between test data and truth data. 
model, weights, RMSE, R2 = MV_linear_regression(file_path, plot_images=True, save_plot=True)