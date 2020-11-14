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
... a GUI will appear (see the image below), within which, the user should select the 2-D data set for which they wish to calculate a linear line of best fit. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/linear-regression/img/File%20selection.PNG?raw=true" alt="file selection GUI" width="500"/>

You can find the example data set within [this folder](https://github.com/SamHSoftware/Machine-Learning/tree/main/linear-regression/data). The first column of the data is considered to be the X data. The second column is considered to be the Y data. 

The data that you need to input must be of two columns, and must be stored within a .csv file.

(4) Upon loading in your data, you may use the following code to calculate accurate values of 'm' and 'c', whilst also saving a graph of the data with the new line of best fit. 
```
# A function to compare the effects of different image filters.
# Function inputs args 1: file_path --> Input as string. The file path for the data in question.
# Function inputs args 2: plot_images --> Set to True or False. Wehn True, prints training data with regression line to console.
# Function inputs args 3: save_plot --> Set to True or False. When True, saves training data to file_path folder.
# Function output 1: The coefficient of gradient ('m'). 
# Function output 2: The y intercept ('c'). 
m, c = linear_regression(file_path, plot_images, save_plot)
```

The code will then output/save the following style of graph, depending on the input arguments you choose. 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/linear-regression/img/output_graph.png?raw=true" alt="montage of filter effects" width="500"/>

If the graph is saved, it will be saved to the same directory as the data which you originally selected. 

## Testing the ```linear_regression()``` function: 

(1) Simply navigate to the ```tests``` directory, open```test_linear_regression_functions.py``` and run the ```.py``` file.

The unit testing checks to see if: 
- The data is read in correctly. 
- The value of 'm' is correctly calculated when using test data. 
- The value of 'c' is correctly calculated when using test data. 

If there are errors, then the code will print assertion errors explaining what went wrong.  

If there were no errors, then a message confirming this will be printed. 
