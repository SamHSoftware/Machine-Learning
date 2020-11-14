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

(1) Run the ```multivariate_linear_regression_functions.py``` file.  

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
... a GUI will appear (see the image below), within which, the user should select the 2-D data set for which they wish to calculate a linear line of best fit. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/multivariate-linear-regression/img/File%20selection.PNG?raw=true" alt="file selection GUI" width="500"/>

You can find the example data set within [this folder](https://github.com/SamHSoftware/Machine-Learning/tree/main/linear-regression/data). The first column of the data is considered to be the X data. The second column is considered to be the Y data. 

The data that you need to input must be of two columns, and must be stored within a .csv file.

