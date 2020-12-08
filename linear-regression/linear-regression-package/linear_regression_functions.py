from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd 
import ntpath
import os

# A function to allow the user to select the image they wish to analyse. 
# Function inputs args: None. 
# Function output 1: The file path of that which was selected by the user. 
def file_selection_dialog():
    root = Tk()
    root.title('Please select the file in question')
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=[("All files", "*.*")])
    file_path = root.filename
    root.destroy()

    return 

# A function to train a linear model, such that it can make predictions given similar data.
# Function inputs args 1: file_path --> Input as string. The file path for the data in question.
# Function inputs args 2: plot_images --> Set to True or False. When True, prints training data with regression line to console.
# Function inputs args 3: save_plot --> Set to True or False. When True, saves training data to file_path folder.
# Function output 1: The coefficient of gradient ('m'). 
# Function output 2: The y intercept ('c'). 
def linear_regression(file_path, plot_images, save_plot): 
    
    # Import the csv file. 
    data = pd.read_csv(file_path, dtype=float)
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    
    # Build the linear regression model. 
    m = 0 # 'm' represents the gradient. 
    c = 0 # 'c' represents the y intercept. 
    a = 0.0001  # The learning rate: Rate at which the suggested line of best fit is altered each iteration. If this value is too great, then instead of converging towards alues of m and c which give the best line of fit, we may diverge from it. 
    iterations = 1000  # The number of iterations to perform gradient descent.

    n = len(X) # Number 'truth' data points from our .csv file. 

    # Performing Gradient Descent 
    for i in range(iterations): 
        predicted_values = m*X + c  # The current predicted value of Y
        
        new_m = (1/n) * sum(X * (predicted_values - Y))  
        new_c = (1/n) * sum(predicted_values - Y)
        
        # Here, we update the values of m and c. It is important to perform this update after the derivatives of *both* have been taken. 
        m = m - a * new_m  # Update m.
        c = c - a * new_c  # Update c.

    # Making predictions
    predicted_values = m*X + c
    
    # Plot the original data with the new linear regression line.
    plt.scatter(X, Y) 
    plt.plot([min(X), max(X)], [min(predicted_values), max(predicted_values)], color='red')  
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Y data', labelpad=10) # The leftpad argument alters the distance of the axis label from the axis itself. 
    plt.xlabel('X data', labelpad=10)
    plt.title('Training data and linear line of best fit', pad=15)
    
    # Save the plot if the user desires it.
    if save_plot:
        _, tail = ntpath.split(file_path)
        new_file_path = file_path.replace('csv', 'png')
        plt.savefig(new_file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (plot_images == False):
        plt.close()
    else:
        plt.show()    
    
    return (m, c)