# data loader
from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2

# Function to load the cifar10 data and add labels to the data (as it comes without labels)
def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # built in function to load data
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    return X_train, y_train, X_test, y_test, labels

# Function to convert the data to grey scale and scale it
def convert_to_grey_scale(X_train, X_test):
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train]) # convert each image in X_train to grey scale using list comprehension
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    return X_train_scaled, X_test_scaled

# Function to flatten the data (both train and test) using reshape
def flatten_data(X_train_scaled, X_test_scaled):
    # For training data
    nsamples, nx, ny = X_train_scaled.shape # returns (50000, 32, 32)
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny)) # Here we are reshaping the data to be 50000 rows, and 1024 columns.
    # For testing data
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train_dataset, X_test_dataset