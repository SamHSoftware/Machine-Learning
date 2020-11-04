# README for project: linear-regression

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com
git commit . -m "README.md update"

## Description:   
- This purpose of this package is to allow a user to fit a linear line of best fit to a 2D data set.
- The user selects the data in question, and a cost function/gradient descent algorithm is used to calculate suitable values of 'm' (the coefficient of gradient) and 'c' (the y-intercept). 
  - To find out more about gradient descent and cost functions, I'd recommend visiting [Andrew NG's machine learning course](https://www.coursera.org/learn/machine-learning/home/welcome). 
- The code will save a graph displaying the 2D data with the linear line of best fit. 
- The code will also output values of 'm' and 'c'. 
- For each ```.py``` file provided, there is an accompanying ```.ipynb``` file for those who use JupiterLab.

## How to use the ```RUNME.py``` code: 

(1) First, open the ```RUNME.py``` file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code: 

```
# Import the necessary packages.
from linear_regression_functions import *
```

(3) Then, with the following function...
```
# A function to allow the user to select the image they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the image in question. 
file_path = file_selection_dialog()
```