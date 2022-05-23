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
def test_load_data():
    
    # Deduce the location of the sentiment labelled sentences. 
    current_directory = os.getcwd()
    current_directory = current_directory.replace('tests', 'training-data')
    data_directory = os.path.join(current_directory, 'sentiment labelled sentences')
    
    # Load the data. 
    df = load_data(data_directory)
    
    # Run assertion tests. 
    assert list(df.columns) == ['sentence', 'label', 'company'], 'Test 1.1 failed. Column names of DataFrame not as expected.'
    assert df.iloc[0,0] == 'Wow... Loved this place.', 'Test 1.2 failed. First review not as expected.'
    assert df.iloc[-1,0] == "All in all its an insult to one's intelligence and a huge waste of money.  ", 'Test 1.2 failed. Last review not as expected.'

    print('Test complete. No errors detected')