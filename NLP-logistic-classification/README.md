# NB: This project is not yet complete. 

# README for project: NLP-logistic-classification

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com  
Date created: 19<sup>th</sup> May 2022

## Description:   
- This code serves as a simple example of natural language processing (NLP), a highly useful field which employs machine and deep learning techniques to analyse text and speech. 
- This code considers text-based reviews which have been collected online, each of which has a corresponding binary label to denote the sentiment of the review, 'positive' or 'negative'. 
- These texts are converted to numerical vectors, to create aBag Of Words (BOW) model. 
- The dataset in question is the (Sentiment Labelled Sentences Data Set, which is provided by the UCI Machine Learning Repository)[https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences].
- Once trained (instructions below), the model will be able to classify text based inputs, and produce a confusion matric to allow model evaluation. 

## Package architecture:
- NLP-logistic-regression:
    * This is where the main code is stored.
    * There are module files (these contain the individual functions) and RUNME files (these actually call (use) the functions).
    * There are .py files for general purpose IDEs, and .ipynb files for JupyterLab.
- training-data: 
    * This folder contains the Sentiment Labelled Sentences Data set, in case you can't find it via the link above. 
- tests:
    * The test file to assess functions within ```code_module.py```.
- img:
    * This is where images are stored. These images are used to illustrate this README document.
- LICENCE.txt:
    * The licence explaining how this code can be used.
- README.md:
    * The file which creates the README for this code.
- environment.yml:
    * A file to allow you to re-create this code's environment in conda (if you're using it). 
- requirements.txt:
    * A file to allow you to re-create this code's environment using pip.

## Here's how to unit test the package before using it: 

(1) Open `RUNME_to_test_code_module.py`. 

(2) Run the code; it will perform unit testing. The testing is within the file is a proof of concept, and shouldn't be viewed as an exhaustive list of tests. 

(3) If there are errors, explanatory print statements will be created. If no errors are detected, a corresponding message will be printed. 

## How to train and evaluate the model. 

(1) Open ```RUNME_to_train_model.py```. 

(2) Select and run the following code: 

```
# Function to train the model. 
# Function output 1: A confusion matrix, saved to a new directory 'img' within your directory of training data.
train_model()
```

(3) A GUI will appear, asking you to select the folder containing the sentiment labelled sentence `.txt` files. The GUI will look like this: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/NLP-logistic-classification/img/folder_selection.PNG?raw=true" alt="An example of the folder selection GUI." width="500"/>  

(4) The model will train (probably pretty quickly, as it's quite simple), and will then output a confusion matrix denote the true/false positive/negative classifictions. The confusion matrix will look like this: 

<img src="https://github.com/SamHSoftware/Machine-Learning/blob/main/NLP-logistic-classification/img/confusion_matrix.png?raw=true" alt="An example confusion matrix used for model evaluation." width="500"/> 

The confusion matrix will be saved to a new directory, named `img`, within the `sentiment labelled sentences`. 

As you can see, considering the simplicity of the model, it seems to be training relativly well. 