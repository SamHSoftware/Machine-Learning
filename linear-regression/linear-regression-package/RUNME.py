# Import the necessary packages.
from linear_regression_functions import *

# A function to allow the user to select the data they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the file in question. 
file_path = file_selection_dialog()

# A function to compare the effects of different image filters.
# Function inputs args 1: file_path --> Input as string. The file path for the data in question.
# Function inputs args 2: plot_images --> Set to True or False. Wehn True, prints training data with regression line to console.
# Function inputs args 3: save_plot --> Set to True or False. When True, saves training data to file_path folder.
# Function output 1: The coefficient of gradient ('m'). 
# Function output 2: The y intercept ('c'). 
m, c = linear_regression(file_path, plot_images, save_plot)