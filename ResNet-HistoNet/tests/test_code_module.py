# These tests present an example of those which could be emplyed, and are by no means exhaustive. 

# Import the necessary packages and modules. 
import os
current_directory = os.getcwd()
module_directory = current_directory.replace('tests', 'ResNet-HistoNet')
import sys
sys.path.insert(0, module_directory)
from code_module import *

# Test the function 'linear_regression()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_code_module(): 

    # Reconstruct the path so that we can load in the provided .csv data from the 'data' folder. 
    folder_path = current_directory.replace('tests', 'training-data')
                            
    # Load extract information from the csv file using get_names_and_distributions.
    image_paths, distributions = get_names_and_distributions(folder_path)
    distributions = np.around(distributions, decimals=3)
    
    # Test 1: Test to see if the data has loaded in correctly.
    if data is None: 
        raise TypeError("Test 1 failed. Error opening .csv. 'area-data.csv' loaded as type None") 
    
    image_paths_truth = ['C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_0.tif',
                          'C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_1.tif',
                          'C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_2.tif',
                          'C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_3.tif',
                          'C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_4.tif',
                          'C:/Users/Samuel Huguet/OneDrive/Documents/Personal/Python/Machine-Learning/ResNet-HistoNet\\training-data\\image_5.tif']

    distributions_truth = np.array([[0.        , 0.        , 0.33333333, 0.22222222, 0.44444444],
                                    [0.55555556, 0.44444444, 0.        , 0.        , 0.        ],
                                    [0.55555556, 0.44444444, 0.        , 0.        , 0.        ],
                                    [0.22222222, 0.55555556, 0.22222222, 0.        , 0.        ],
                                    [0.        , 0.66666667, 0.33333333, 0.        , 0.        ],
                                    [0.        , 0.55555556, 0.22222222, 0.22222222, 0.        ]])
    distributions_truth = np.around(distributions_truth, decimals=3)
    
    # Test 2: See if the contents of the csv file match that which we expect. 
    assert image_paths == image_paths_truth, "Test 2.1 failed. Path information within csv file did not matchoutput of get_names_and_distributions(csv_path)"
    assert np.all(distributions == distributions_truth), "Test 2.2 failed. Distribution information within csv file did not match output of get_names_and_distributions(csv_path)"
    
    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_code_module()