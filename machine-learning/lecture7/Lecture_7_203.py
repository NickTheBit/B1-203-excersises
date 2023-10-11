# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 11/10/2023
# Lecture: 7 Support vector machines
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.11.4
# Functionality: 
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

# Create and fit SVM on training data
svm = SVC(C=1.0, kernel='rbf', gamma='scale')
svm.fit(train_data, train_labels)

# Load the test data
test_data = np.concatenate(
    [mnist[f"test{i}"] for i in range(num_classes)]) / 255.0
test_labels = np.concatenate(
    [np.full(len(mnist[f"test{i}"]), i) for i in range(num_classes)])

# Test model on test set
predictions_svm = svm.predict(test_data)

# Calculate accuracy on the test set for SVM
accuracy_svm = accuracy_score(test_labels, predictions_svm)
print(f"Accuracy on the test set for SVM: {accuracy_svm:.4f}")

# Plot Confusion matrix for SVM
confusion_matrix_svm = confusion_matrix(
    test_labels, predictions_svm, labels=np.arange(num_classes))
ConfusionMatrixDisplay(confusion_matrix_svm, display_labels=np.arange(
    num_classes)).plot(cmap='Blues', values_format='d')
plt.title("SVM Confusion Matrix")
plt.savefig("assets/figure1.png")

# Perform PCA
pca = PCA(n_components=2)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)

# Create and fit SVM on PCA-transformed training data
svm_pca = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_pca.fit(train_data_pca, train_labels)

# Test model on PCA-transformed test data
predictions_svm_pca = svm_pca.predict(test_data_pca)

# Calculate accuracy on the test set for SVM with PCA
accuracy_svm_pca = accuracy_score(test_labels, predictions_svm_pca)
print(f"Accuracy on the test set for SVM with PCA: {accuracy_svm_pca:.4f}")

# Plot Confusion matrix for SVM with PCA
confusion_matrix_svm_pca = confusion_matrix(
    test_labels, predictions_svm_pca, labels=np.arange(num_classes))
ConfusionMatrixDisplay(confusion_matrix_svm_pca, display_labels=np.arange(
    num_classes)).plot(cmap='Blues', values_format='d')
plt.title("SVM with PCA Confusion Matrix")
plt.savefig("assets/firugre2.png")

# Perform LDA
lda = LDA(n_components=2)
lda.fit(train_data, train_labels)
train_data_lda = lda.transform(train_data)
test_data_lda = lda.transform(test_data)

# Create and fit SVM on LDA-transformed training data
svm_lda = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_lda.fit(train_data_lda, train_labels)

# Test model on LDA-transformed test data
predictions_svm_lda = svm_lda.predict(test_data_lda)

# Calculate accuracy on the test set for SVM with LDA
accuracy_svm_lda = accuracy_score(test_labels, predictions_svm_lda)
print(f"Accuracy on the test set for SVM with LDA: {accuracy_svm_lda:.4f}")

# Plot Confusion matrix for SVM with LDA
confusion_matrix_svm_lda = confusion_matrix(
    test_labels, predictions_svm_lda, labels=np.arange(num_classes))
ConfusionMatrixDisplay(confusion_matrix_svm_lda, display_labels=np.arange(
    num_classes)).plot(cmap='Blues', values_format='d')
plt.title("SVM with LDA Confusion Matrix")
plt.savefig("assets/figure3.png")