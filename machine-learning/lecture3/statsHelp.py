# File containing all the statistics calculations needed in one place.
# so I don't have to look at them any more than I need to.

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf

def getMean(dataset):
	return np.mean(dataset, axis=0)

def getCovariance(dataset):
	return np.cov(dataset, rowvar=False)

def likelihood(X, mean, covariance):
	n, d = X.shape
	if d != 2:
		raise ValueError("Only 2D data points are allowed.")

	# Calculate the constant term
	constant = 1.0 / (2 * np.pi * np.linalg.det(covariance) ** 0.5)

	# Initialize the result array
	result = np.zeros(n)

	# Invert the covariance matrix once to save time
	Sigma_inv = np.linalg.inv(covariance)

	for i in range(n):
		x = X[i, :]
		
		# Calculate the exponent term
		exponent_term = np.exp(-0.5 * np.dot(np.dot((x - mean).T, Sigma_inv), (x - mean)))
		
		result[i] = constant * exponent_term

	
	likelihood_value = result
	return likelihood_value
