# README for project: GAN_expand_nuclear_dataset

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description: 

## Package architecture:
- GAN_expand_nuclear_dataset:
    * This is where the main code is stored.
    * There are module files (these contain the individual functions) and RUNME files (these actually call (use) the functions).
    * There are .py files for general purpose IDEs, and .ipynb files for JupyterLab.
- img:
    * This is where images are stored. These images are used to illustrate this README document.
- .gitignore:
    * I use software called 'Git Bash' to sync my local files with those within this GitHub repositry. The .gitignore file contains a list of directories (e.g. text files with project notes), and prevents Git Bash from uplodaing them; I don't want to clutter up this repo!
- LICENCE.txt:
    * The licence explaining how this code can be used.
- README.md:
    * The file which creates the README for this code.
- environment.yml:
    * A file to allow you to re-create this code's environment in conda.
- requirements.txt:
    * A file to allow you to re-create this code's environment using pip.

## Requirements. 
(1) Please see the ```requirements.txt``` file (or ```environment.yml``` file if you are using conda) to note the packages (and their respective versions) which are needed for this code to run. 

## How to use this code to classify pixels into different classes. 

### (1) Preparing your data for training. 

### (2) Selecting the data. 

Open, ```RUNME.py```. Here, you can select and run everything at all once if you're confident. If this is your first time, I'd recommend running the code piece by piece. First run the following code: 

```
code
```

A GUI will appear (see the example below), with which you should select the folder containing your training dataset. 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Question_mark_%28black%29.svg/800px-Question_mark_%28black%29.svg.png" alt="An example of the GUI used to select training dataset directory" width="500"/>

### (3) Training your model.

Consider the following code: 

```
code
```

### (4) Continuing the training for a previous model. 


### (5) Use your trained model to ...

Use the following code: 

```
code
```
