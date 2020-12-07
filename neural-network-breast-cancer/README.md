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

