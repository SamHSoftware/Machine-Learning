# README for project: multivariate-linear-regression

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com  
Date created: 30<sup>th</sup> November 2020

## Description:   
- The purpose of this package is to allow a user to train a neural network with multidimensional data to classify breast cancers as malignant or benign. 
- This data can be easily swapped out for your own data. If you're unsure as to how you might do this, take some guidance from [this multivariate-linear regression-package that I wrote](https://github.com/SamHSoftware/Machine-Learning/tree/main/multivariate-linear-regression). There, you can find functions which allow you to load in your multivariate data.
- The neural netowrk is constructed using the PyTorch package. 
- The code will provide a number of different outputs, including the trained model, the predicted values, the corresponding truth values, a confusion matrix, training and validation accuracy per epoch and training and validation loss per epoch. 


## Here's how to unit test the package before using it: 

(1) Run the ```test_neural_network_functions.py``` file.  

This code will automatically check to see if the Pytorch model achieves the expected accuracy after 10000 training epochs. If an error is detected, the code will notify you of the error and will give a description of what has gone wrong. If no errors are detected, then the code will print a statement confirming this, and the rest of the package will be good to run. 

## How to use the ```RUNME.py``` file and use the package: 

(1) Open the ```RUNME.py``` file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code:

```
# Import the necessary packages.
from neural_netowrk_functions import *
```

(3) Then with the following function... 
```
# A function which trains itself to predict whether a cancer is metastatic or not by using a neural netowrk. 
# Function inputs arg 1: save_plot --> True or False. When True, saves plot to the img folder of the package. 
# Function inputs arg 2: display_plot --> True or False. When True, displays plot within conole. 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Predicted values --> Outputs a list of predicted values.  
# Function output 3: True values --> Outputs a list of True values corresponding to the predicted values.
neural_network(save_plot=True, 
               display_plot=True): 
```

... a medical data set will be split into testing and training groups. The training data will be used to train the neural network. During the training process, the training and validation loss and accuracy will be calculated and recorded. At the end of the training process, graphs of each will be plotted. You can see examples below: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/neural-network-breast-cancer/img/training_and_validation_accuracy.png?raw=true" alt="Validation and training accuracy per epoch" width="500"/>  

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/neural-network-breast-cancer/img/training_and_validation_loss.png?raw=true" alt="Validation and training loss per epoch" width="500"/>  

In addition, a confusion matrix will be produced to help you discern how well your model functions. You can see an example of the confusion matrix below: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/neural-network-breast-cancer/img/confusion_matrix.png?raw=true" alt="Validation and training loss per epoch" width="500"/>  

If the graphs are saved, it will be saved to the same directory as the data which you originally selected.

