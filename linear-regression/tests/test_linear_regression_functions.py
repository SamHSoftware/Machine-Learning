# Import the necessary packages and modules. 
import numpy as np 
import pandas as pd 
import os
current_directory = os.getcwd()
module_directory = current_directory.replace('tests', 'linear-regression-package')
import sys
sys.path.insert(0, module_directory)
from linear_regression_functions import *

# Test the function 'linear_regression()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_linear_regression(): 

    # Reconstruct the path so that we can load in the provided .csv data from the 'data' folder. 
    file_path = current_directory.replace('tests', 'data/linear_regression_data.csv')

    # Load in the csv data. 
    data = pd.read_csv(file_path, dtype=float)    

    # Test 1: Test to see if the data has loaded in correctly.
    if data is None: 
        raise TypeError("Test 1 failed. Error opening .csv.'linear_regression_data.csv' is of type None") 
    
    # Test 2: See if the values of 'm' and 'c' (from our linear regression function) match the expected values. 
    m_test, c_test = linear_regression(file_path, plot_images=False, save_plot=False)
    assert m_test == 0.9873951622984585, "Test 2.1 failed. The calculated coefficient of gradient is not equal to 0.9873951622984585"
    assert c_test == 0.1288261344615331, "Test 2.2 failed. The calculated y-intercept is not equal to 0.1288261344615331"
    
    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_linear_regression()