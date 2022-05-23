# This file serves as an example of tests that could be performed, 
# and is not reflective of the degree of test that such functions 
# would require. 

# Import the necessary packages and modules. 
import os
current_directory = os.getcwd()
module_directory = current_directory.replace('tests', 'NLP-logistic-classification')
import sys
sys.path.insert(0, module_directory)
from code_module import *

# Function to test load_data(). 
test_load_data()