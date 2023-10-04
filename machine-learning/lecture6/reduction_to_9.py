# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 04/11/2023
# Lecture: Linear discriminants
# Dependencies: numpy, matplotlib, sklearn, scipy
# Python version: 3.11.4
# Functionality: Classification, this one reduces the dataset to 9 dimentions.
# ###################################


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the MNIST dataset
mnist = loadmat("mnist_all.mat")

# Concatenate the training data from all digit classes
train_data = np.concatenate([mnist[f"train{i}"] for i in range(10)], axis=0) / 255.0

# Create labels for the training data
labels = np.concatenate([i * np.ones(len(mnist[f"train{i}"])) for i in range(10)])

# Define the number of dimensions for dimensionality reduction (2 for PCA, 2 for LDA)
num_dimensions = 9

# Perform dimensionality reduction using PCA
pca = PCA(n_components=num_dimensions)
train_data_pca = pca.fit_transform(train_data)

# Perform dimensionality reduction using LDA
lda = LDA(n_components=num_dimensions)
train_data_lda = lda.fit_transform(train_data, labels)

# Train a classifier (e.g., k-Nearest Neighbors) using PCA-reduced data
from sklearn.neighbors import KNeighborsClassifier

knn_pca = KNeighborsClassifier(n_neighbors=9)
knn_pca.fit(train_data_pca, labels)

# Train a classifier using LDA-reduced data
knn_lda = KNeighborsClassifier(n_neighbors=9)
knn_lda.fit(train_data_lda, labels)

# Load the test data
test_data = np.concatenate([mnist[f"test{i}"] for i in range(10)], axis=0) / 255.0
test_labels = np.concatenate([i * np.ones(len(mnist[f"test{i}"])) for i in range(10)])

# Perform dimensionality reduction on test data using PCA and LDA
test_data_pca = pca.transform(test_data)
test_data_lda = lda.transform(test_data)

# Make predictions using PCA-based k-NN
predictions_pca = knn_pca.predict(test_data_pca)

# Make predictions using LDA-based k-NN
predictions_lda = knn_lda.predict(test_data_lda)

# Calculate and print accuracy scores for both PCA and LDA
accuracy_pca = accuracy_score(test_labels, predictions_pca)
accuracy_lda = accuracy_score(test_labels, predictions_lda)
print(f"Accuracy using PCA: {accuracy_pca:.2f}")
print(f"Accuracy using LDA: {accuracy_lda:.2f}")

# Create confusion matrices and display them
confusion_matrix_pca = confusion_matrix(test_labels, predictions_pca)
confusion_matrix_lda = confusion_matrix(test_labels, predictions_lda)

# plt.figure(figsize=(12, 4))


pcaPlot = ConfusionMatrixDisplay(confusion_matrix_pca, display_labels=np.arange(10))
pcaPlot.plot(cmap='Blues', values_format='d').ax_.set_title("PCA Reduction to 9 dimentions")
plt.savefig("pcaReduction_to9.png")

ldaPlot = ConfusionMatrixDisplay(confusion_matrix_lda, display_labels=np.arange(10))
ldaPlot.plot(cmap='Blues', values_format='d').ax_.set_title("LDA Reduction to 9 dimentions")
plt.savefig("ldaReduction_to9.png")