# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 13/09/2023
# Lecture: 4 Dimensionality reduction
# Dependencies: numpy, matplotlib, sklearn, scipy
# Python version: 3.11.4
# Functionality: Excersise on dimention reduction with both PCA 
# and LDA methods, and comparison of the generated groups.
# ###################################

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal as norm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Loading and normalizing training data
train5 = np.loadtxt("mnist_all_ASCII/mnist_all/train5.txt") / 255 # /255 for normalization
train6 = np.loadtxt("mnist_all_ASCII/mnist_all/train6.txt") / 255
train8 = np.loadtxt("mnist_all_ASCII/mnist_all/train8.txt") / 255

# Defining targets, what does that mean?
train5_target = 5*np.ones(len(train5))
train6_target = 6*np.ones(len(train6))
train8_target = 8*np.ones(len(train8))

# Combining all data on a single dataset.
train_data = np.concatenate([train5, train6, train8])
train_targets = np.concatenate([train5_target, train6_target, train8_target])


# Loading and normalizing testing data
test5 = np.loadtxt("mnist_all_ASCII/mnist_all/test5.txt") / 255
test6 = np.loadtxt("mnist_all_ASCII/mnist_all/test6.txt") / 255
test8 = np.loadtxt("mnist_all_ASCII/mnist_all/test8.txt") / 255

# Define targets 
test5_target = 5*np.ones(len(test5))
test6_target = 6*np.ones(len(test6))
test8_target = 8*np.ones(len(test8))

# Combine
test_data = np.concatenate([test5, test6, test8])
test_targets = np.concatenate([test5_target, test6_target, test8_target])

# Class names
classes = np.array([5,6,8])

# Verify the following line manipulates the data as expected.
# # Part 1: Reduce dimension to 2
# Here, we wish to reduce the data dimensionality from 784 to 2 using either PCA or LDA.
# For this you can use scikit-learn.
# Reducing the dimentions of the dataset using PCA
pca = PCA(n_components=2)


# The PCA class in scikit-learn fits a covariance matrix and compute eigenvectors for you. 
# PCA doesn't assume any knowledge about the classes, so you have to use the concatenated training set.


# Fit a scikit learn PCA instance to training data
principal_components = pca.fit(train_data)


# Now that the PCA model is fit to the training data, we can find a low dimesional representation of each class.
train5_transformed_PCA = principal_components.transform(train5)
train6_transformed_PCA = principal_components.transform(train6)
train8_transformed_PCA = principal_components.transform(train8)

# ## LDA
# We can also use Linear Disicriminant Analysis to reduce the dimensionality of the data.
# The LDA class in scikit-learn fits a covariance matrix and compute eigenvectors for you. LDA assume that you 
# know about the classes, so you have to use the concatenated training set and targets/classes

# Fit a scikit learn LDA instance to training data
lda = LDA(n_components=2)
train_lda = lda.fit(train_data, train_targets)

# Transform train data from each class using fitted LDA instance
train5_transformed_LDA = train_lda.transform(train5)
train6_transformed_LDA = train_lda.transform(train6)
train8_transformed_LDA = train_lda.transform(train8)


# Let's try to plot the dimensionality reduced data and compare PCA to LDA. What do we see?
#Scatter plot of the dimensional-reduced data 
plt.scatter(train5_transformed_LDA[:, 0], train5_transformed_LDA[:, 1], color='navy', marker='x', s=0.2)
plt.scatter(train6_transformed_LDA[:, 0], train6_transformed_LDA[:, 1], color='turquoise', marker='+', s=0.2)
plt.scatter(train8_transformed_LDA[:, 0], train8_transformed_LDA[:, 1],color='darkorange', marker='*', s=0.2)
# plt.show()

plt.scatter(train5_transformed_PCA[:, 0], train5_transformed_PCA[:, 1], color='navy', marker='x', s=0.2)
plt.scatter(train6_transformed_PCA[:, 0], train6_transformed_PCA[:, 1], color='turquoise', marker='+', s=0.2)
plt.scatter(train8_transformed_PCA[:, 0], train8_transformed_PCA[:, 1],color='darkorange', marker='*', s=0.2)
# plt.show()


# In the above plot we see that LDA is seemingly better at seperating the tree classes,while the classes 5 and 8 are highly overlapped when using PCA.


# # Part 2: Perform 3-class classification based on the generated 2-dimensional data. 
# We need to find a model to classify the test data as either 5, 6, or 8.
# Here, we could use a Gaussian model for each class, and estimate the mean and covariance from the dimensionality reduced data.

# %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from PCA

# %%
#Estimate parameters for a bivariante Gaussian distribution.
def compute_mean_covariance(data):
    # statistics
    mean = np.mean(data, axis=0) # To calcuate the mean from two dimentional data set np.mean(data, axis=0)
    covariance= np.cov(data, rowvar=False) # To calculate the covariance matrix np.cov(data, rowvar=False)

    return mean, covariance

train_5_mean_pca,train_5_cov_pca=compute_mean_covariance(train5_transformed_PCA)
train_6_mean_pca,train_6_cov_pca=compute_mean_covariance(train6_transformed_PCA)
train_8_mean_pca,train_8_cov_pca=compute_mean_covariance(train8_transformed_PCA)

# %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from LDA

# %%
#Estimate parameters for a bivariante Gaussian distribution.
train_5_mean,train_5_cov=compute_mean_covariance(train5_transformed_LDA)
train_6_mean,train_6_cov=compute_mean_covariance(train6_transformed_LDA)
train_8_mean,train_8_cov=compute_mean_covariance(train8_transformed_LDA)

# Transform test data using fitted PCA/LDA instance



test_data_reduced_pca = principal_components.transform(test_data)
test_data_reduced_lda = train_lda.transform(test_data)


# %% [markdown]
# Now we compute priors, likelihoods and posteriors

# %%
# Compute priors
prior_5 = (len(train5))/(len(train5)+len(train6)+len(train8))
prior_6 = (len(train6))/(len(train5)+len(train6)+len(train8))
prior_8 = (len(train8))/(len(train5)+len(train6)+len(train8))

# Compute Likelihoods
def likelihood(X, mu, Sigma):
    n, d = X.shape
    if d != 2:
        raise ValueError("Only 2D data points are allowed.")

    # Calculate the constant term
    constant = 1.0 / (2 * np.pi * np.linalg.det(Sigma) ** 0.5)

    # Initialize the result array
    result = np.zeros(n)

    # Invert the covariance matrix once to save time
    Sigma_inv = np.linalg.inv(Sigma)

    for i in range(n):
        x = X[i, :]
        
        # Calculate the exponent term
        exponent_term = np.exp(-0.5 * np.dot(np.dot((x - mu).T, Sigma_inv), (x - mu)))
        
        result[i] = constant * exponent_term

    
    likelihood_value = result
    return likelihood_value

data_lda=test_data_reduced_lda
likelihood_5_lda= likelihood(data_lda,train_5_mean,train_5_cov)
likelihood_6_lda= likelihood(data_lda,train_6_mean,train_6_cov)
likelihood_8_lda= likelihood(data_lda,train_8_mean,train_8_cov)

data_pca=test_data_reduced_pca
likelihood_5_pca= likelihood(data_pca,train_5_mean_pca,train_5_cov_pca)
likelihood_6_pca= likelihood(data_pca,train_6_mean_pca,train_6_cov_pca)
likelihood_8_pca= likelihood(data_pca,train_8_mean_pca,train_8_cov_pca)
# Compute posteriors
posterior_5_lda= (likelihood_5_lda*prior_5)
posterior_6_lda= (likelihood_6_lda*prior_6)
posterior_8_lda= (likelihood_8_lda*prior_8)

posterior_5_pca= (likelihood_5_pca*prior_5)
posterior_6_pca= (likelihood_6_pca*prior_6)
posterior_8_pca= (likelihood_8_pca*prior_8)

# %% [markdown]
# We can now compute the classification accuracy for both PCA and LDA

#Compute predictions
classifications_lda = np.where(posterior_5_lda > posterior_6_lda, np.where(posterior_5_lda > posterior_8_lda, 5, 8), np.where(posterior_6_lda > posterior_8_lda, 6, 8))
print(f"The classifications for the test dataset using lda are {classifications_lda}.")
count_correct_lda = np.count_nonzero(classifications_lda == test_targets)

classifications_pca = np.where(posterior_5_pca > posterior_6_pca, np.where(posterior_5_pca > posterior_8_pca, 5, 8), np.where(posterior_6_pca > posterior_8_pca, 6, 8))
print(f"The classifications for the test dataset using pca are {classifications_pca}.")
count_correct_pca = np.count_nonzero(classifications_pca == test_targets)
#Compute accuracy
accuracy= count_correct_pca/len(classifications_pca)
print(f"The accuracy (pca) is {accuracy}.")

accuracy= count_correct_lda/len(classifications_lda)
print(f"The accuracy (lda) is {accuracy}.")

