# Import the necessary packages and modules. 
import os
current_directory = os.getcwd()
module_directory = current_directory.replace('tests', 'logistic-classification-package')
import sys
sys.path.insert(0, module_directory)
from logistic_classification_functions import *

# Test the function 'logistic_classification()' against the provided outputs. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_logistic_classification(): 

    _, accuracy, y_predicted_classes, y_testing = logistic_classification(False, False)
    
    # Test 2: See if the values of  RMSE, R2 and the weigths for the linear model fall within the expeted range. 
    assert accuracy == 0.9680851063829787, "Test 1 failed. The calculated value accuracy for the model(using a fixed data set) was not equal to 0.9680851063829787"

    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_logistic_classification()