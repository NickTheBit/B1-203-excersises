# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 27/09/2023
# Lecture: 5 Clustering
# Dependencies: numpy, matplotlib, scipi
# Python version: 3.11.4
# Functionality: 
# ###################################

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

# Load the new dataset
data_path = "2D568class.mat"
data = loadmat(data_path)
train5_2dim = data["trn5_2dim"] / 255
train6_2dim = data["trn6_2dim"] / 255
train8_2dim = data["trn8_2dim"] / 255

# Load the old dataset (MNIST) and concatenate it with the new dataset
# Assuming you have already loaded and prepared the MNIST dataset as mentioned in your previous code
# Concatenate the old and new training data
train_data_combined = np.concatenate([train5_2dim, train6_2dim, train8_2dim, train_data])

# Perform dimensionality reduction using PCA or LDA as before (you can choose either)
# Here, I'll use PCA for illustration
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_combined)

# Create a Gaussian Mixture Model (GMM) to model the mixed 2-dimensional data
num_components = 3  # Number of components for GMM (3 classes: 5, 6, 8)
gmm = GMM(n_components=num_components)
gmm.fit(train_data_pca)

# Estimate Gaussian parameters for the GMM components
means = gmm.means_
covariances = gmm.covariances_

# Visualize the GMM components
colors = ['navy', 'turquoise', 'darkorange']

plt.figure()
for i, color in enumerate(colors):
    eigenvalue, eigenvector = np.linalg.eigh(covariances[i])
    normalized_eigenvector = eigenvector[0] / np.linalg.norm(eigenvector[0])
    angle = np.arctan2(normalized_eigenvector[1], normalized_eigenvector[0])
    angle = 180 * angle / np.pi
    scaling_factor = 8
    eigenvalue = scaling_factor * eigenvalue
    ell = plt.matplotlib.patches.Ellipse(
        means[i], eigenvalue[0], eigenvalue[1], 180 + angle, color=color
    )
    ell.set_clip_box(plt.gca().bbox)
    ell.set_alpha(0.5)
    plt.gca().add_artist(ell)

# Plot the reduced data points
plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], 0.8)
plt.title('Gaussian Mixture Model for Mixed 2D Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()