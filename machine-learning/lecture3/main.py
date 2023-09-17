# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Lidal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 13/09/2023
# Lecture: 3 Parametric and nonparametric methods
# Dependencies: numpy, matplotlib
# Python version: 3.11.4
# Functionality: 
# ###################################

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf
import statsHelp as sh

# Importing Data
# Trainining data
train_x = np.loadtxt("dataset1_G_noisy_ASCII/trn_x.txt")
train_x_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_x_class.txt")

train_y = np.loadtxt("dataset1_G_noisy_ASCII/trn_y.txt")
train_y_label = np.loadtxt("dataset1_G_noisy_ASCII/trn_y_class.txt")

train_mean = np.mean(train_x, axis= 0)
train_var = np.var(train_x, axis= 0)

# Testing data
test_x = np.loadtxt("dataset1_G_noisy_ASCII/tst_x.txt")
test_x_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_x_class.txt")

test_y = np.loadtxt("dataset1_G_noisy_ASCII/tst_y.txt")
test_y_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_class.txt")

test_y_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_126.txt")
test_y_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_y_126_class.txt")

test_xy = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy.txt")
test_xy_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_class.txt")

test_xy_126 = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126.txt")
test_xy_126_label = np.loadtxt("dataset1_G_noisy_ASCII/tst_xy_126_class.txt")


# figure out the mean and the covariance for our training datasets.
train_x_mean = sh.getMean(train_x)
train_x_covariance = sh.getCovariance(train_x)

train_y_mean = sh.getMean(train_y)
train_y_covariance = sh.getCovariance(train_y)

# Computing the priors
prior_x = sh.getPrior(train_x_label)
prior_y = sh.getPrior(train_y_label)

# This is the likely function for x and y given our training data
def likelihood(data, mean, cov):
    likelihoodValue = 
    return likelihoodValue


# Initial data plot
figure, plotAxises = plt.subplots(figsize=(6,6))

#The correlation can be controlled by the param 'dependency', a 2x2 matrix.
dependency_nstd = [[0.8, 0.75],
                   [-0.2, 0.35]]
mu = 0, 0
scale = 8, 5

# Vertical and horizontal lines for center reference.
plotAxises.axvline(c='grey', lw=1)
plotAxises.axhline(c='grey', lw=1)

x, y = pf.importDataset(test_xy, dependency_nstd, mu, scale)
plotAxises.scatter(x, y, s=0.5, marker='x', c='grey')

pf.confidence_ellipse(x, y, plotAxises, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
pf.confidence_ellipse(x, y, plotAxises, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
pf.confidence_ellipse(x, y, plotAxises, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

plotAxises.scatter(mu[0], mu[1], c='red', s=3)
plotAxises.set_title('Different standard deviations')
plotAxises.legend()
plt.show()