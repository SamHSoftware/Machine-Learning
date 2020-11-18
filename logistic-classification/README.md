# README for project: multivariate-linear-regression

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 
The purpose of this package is to allow a user to fit a logistic regression model to multidimensional data in order to predict a binary output. 
- The mutidimensional data is medical in nature, and classifies whether or not people have breast cancer, while also giving additional lists of quantifiable symptoms which can be used to make useful predictions. 
- This data can be easily swapped out for your own data. If you're unsure as to how you might do this, take some guidance from [this multivariate-linear regression-package that I wrote](https://github.com/SamHSoftware/Machine-Learning/tree/main/multivariate-linear-regression). There, you can find functions which allow you to load in your multivariate data. 
- A PyTorch logistic regression model is trained to predict whether the patients in question do or do not have breast cancer. 
- The code will provide a number of different outputs, including the trained model, the accuracy fo the model, the predicted values and the corresponding truth values. 

## Here's how to unit test the package before using it: 

(1) Run the ```test_logistic_classification_functions.py``` file.  

This code will automatically check to see if the Pytorch model provides the expected output. If an error is detected, the code will notify you of the error and will give a description of what has gone wrong. If no errors are detected, then the code will print a statement confirming this, and the rest of the package will be good to run. 

## How to use the ```RUNME.py``` file and use the package: 

(1) Open the ```RUNME.py``` file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code:

```
# Import the necessary packages.
from logistic_classification_functions import *
```

(3) Then with the following function... 
```
# Function inputs arg 1: display_plot --> Set to True or False. When True, prints graph of BCE calculated loss per epoch.
# Function inputs arg 2: save_plot --> Set to True or False. When True, 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Accuracy --> Outputs the accuracy of the model as a fraction between 0 (innaccurate) and 1 (accurate). 
# Function output 3: Predicted values --> Outputs a list of predicted values.  
# Function output 4: True values --> Outputs a list of True values corresponding to the predicted values.
model, accuracy, predicted_values, true_values = logistic_classification(plot_images, save_plot)
```

... You will be able to predict whether patients have breast cancer or not! The function will also output a graph to shown how the loss dreases with each epoch: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/logistic-classification/img/BCE_calculated_loss.png?raw=true" alt="Loss per epoch" width="500"/>  

If the graphs are saved, it will be saved to the same directory as the data which you originally selected.