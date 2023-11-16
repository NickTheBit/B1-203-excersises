# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 11/10/2023
# Lecture: 
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.11.4
# Functionality: Visualises the data before processing to provide context.
# ###################################

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score


# Load the entire MNIST dataset
mnist = loadmat("mnist_all.mat")

# Define the number of classes (0 to 9)
num_classes = 10

# Create empty arrays to store the data and labels
train_data = np.empty((0, 784))
train_labels = np.empty(0, dtype=int)

# Concatenate the training data from all digit classes
for i in range(num_classes):
    data = mnist[f"train{i}"] / 255.0
    labels = np.full(len(data), i)
    train_data = np.vstack((train_data, data))
    train_labels = np.concatenate((train_labels, labels))

print(data)