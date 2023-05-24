<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Datascience 2023</h1> 
  <h2 align="center">Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks</h2> 
  <h3 align="center">Visual Analytics</h3> 


  <p align="center">
    Jørgen Højlund Wibe<br>
    Student number: 201807750
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
This assignment compares two different methods to classify images based on nine different labels. The two methods are logistic regression and neural networks.
This is done by firstly converting all of the images to gray scale as it helps in simplifying algorithms and as well eliminates the complexities related to computational requirements.
Then we flatten the images to make them from a 3D array to a 1D. 1-D arrays take less memory and thus flattening helps in decreasing the memory as well as reducing the time to train the model. 

After these preprocessing steps, we apply both methods to the data and print a classification report in the end.

<!-- USAGE -->
## Usage

To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.1 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment, install libraries and run the project.

1. Clone repository
2. Run setup.sh

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/AU-CDS/assignment2-image-classification-jorgenhw.git
cd assignment2-image-classification-jorgenhw
```

### Run ```setup.sh```

To replicate the results, I have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```bash
bash setup.sh
```

## Inspecting results

A classification report from the each approach on the data is located in the folder ```out```. Here one can inspect the results.

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:
```
│   main.py
│   README.md
│   requirements.txt
│   setup.sh
│
├───out
│       classification_report_log_reg.csv
│       classification_report_nn.csv
│
└──src
        log_reg.py
        neural_network.py
        preprocessing.py
```


<!-- DATA -->
## Data
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Assignment objectives
You should write code which does the following:

- Load the Cifar10 dataset ✅
- Preprocess the data (e.g. greyscale, reshape) ✅
- Train a classifier on the data ✅
- Save a classification report ✅

## Remarks on findings
This study found that neural networks only slightly outperformed logistic regression at classifying the images.

This suggests that for this particular classification task, the additional complexity and computational resources required to train a neural network may not be worth the modest improvement in accuracy over logistic regression.