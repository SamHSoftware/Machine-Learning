# README for project: multivariate-linear-regression

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description:   
- This purpose of this package is to allow a user to fit a multivariate linear model to multidimensional data in order to predict corresponding output values. 
- The user selects the data in question, stored within a .csv file. 
- A PyTorch model is trained to calculate the weights and biases needed to correctly make predictions from the data.
- The code will save two graphs (shown below) which allow the user to assess the functionality of the trained model, and to visualise the output of the cost function per training iteration. 
- The code also outputs the root mean square error (RMSE) and the R<sup>2</sup> value when comparing model predictions to ground truth data. 
- For each ```.py``` file provided, there is an accompanying ```.ipynb``` file for those who use JupiterLab.

## Here's how to unit test the package before using it: 

(1) Run the ```test_multivariate_linear_regression_functions.py``` file.  

This code will automatically check to see if the Pytorch model outputs the expected weights and biases, RMSE and R<sup>2</sup> value. If an error is detected, the code will notify you of the error and will give a description of what has gone wrong. If no errors are detected, then the code will print a statement confirming this, and the rest of the package will be good to run. 

## How to use the ```RUNME.py``` file and use the package: 

(1) Open the ```RUNME.py``` file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code:

```
# Import the necessary packages.
from multivariate_linear_regression_functions import *
```

(3) Then, with the following function...
```
# A function to allow the user to select the image they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the image in question. 
file_path = file_selection_dialog()
```
... a GUI will appear (see the image below), within which, the user should select the data set which the code is to handle.

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/multivariate-linear-regression/img/File%20selection.PNG?raw=true" alt="file selection GUI" width="500"/>

You can find the example data set within [this folder](https://github.com/SamHSoftware/Machine-Learning/blob/main/multivariate-linear-regression/data/MV_linear_regression_data.csv). The contents of the .csv file need to be organised in the following manner: The first column must be 'truth' values, which we eventually want to learn how to predict. Each successive column must contain the training data, which can be used to train our models and make predictions. Here is an example. If we wanted to predict house prices, then the house prices would go in column 1. If we were using 'floor space' and 'number of rooms' as predictors for house price, then the 'floor space' and 'number of rooms' data would need to reside in columns 2 and 3 respectivly. If you need to use more than 2 predictors, thats fine, just add each new predictor into a new column. The code will recognise their presence and use them for training. 

