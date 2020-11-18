# Import the necessary packages.
from logistic_classification_functions import *

# Function inputs arg 1: display_plot --> Set to True or False. When True, prints graph of BCE calculated loss per epoch.
# Function inputs arg 2: save_plot --> Set to True or False. When True, 
# Function output 1: Model --> Outputs the trained model.
# Function output 2: Accuracy --> Outputs the accuracy of the model as a fraction between 0 (innaccurate) and 1 (accurate). 
# Function output 3: Predicted values --> Outputs a list of predicted values.  
# Function output 4: True values --> Outputs a list of True values corresponding to the predicted values.
model, accuracy, predicted_values, true_values = logistic_classification(plot_images, save_plot)