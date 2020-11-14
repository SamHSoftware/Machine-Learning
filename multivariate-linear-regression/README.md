# README for project: multivariate-linear-regression

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description:   
- This purpose of this package is to allow a user to fit a multivariate linear model to multidimensional data in order to predict corresponding output values. 
- The user selects the data in question, stored within a .csv file. 
- A PyTorch model is trained to calculate the weights and biases needed to correctly make predictions from the data.
- The code will save two graphs (shown below) which allow the user to assess the functionality of the trained model, and to visualise the output of the cost function per training iteration. 
- The code also outputs the root mean square error (RMSE) and the R^2 value when comparing model predictions to ground truth data. 
- For each ```.py``` file provided, there is an accompanying ```.ipynb``` file for those who use JupiterLab.