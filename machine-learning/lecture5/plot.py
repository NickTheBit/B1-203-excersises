import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate some sample data
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(5, 1, 1000)]).reshape(-1, 1)

# Fit a Gaussian Mixture Model to the data
gmm = GaussianMixture(n_components=2)
gmm.fit(X)

# Create a range of values for plotting
x = np.linspace(-5, 10, 1000)
x = x.reshape(-1, 1)

# Compute the log-likelihood of each point
log_probs = gmm.score_samples(x)

# Plot the data and the GMM
plt.scatter(X, np.zeros_like(X), alpha=0.5)
plt.plot(x, np.exp(log_probs), '-r', label='GMM')
plt.xlabel('X-axis')
plt.ylabel('Density')
plt.legend()
plt.show() 