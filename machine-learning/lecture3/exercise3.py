# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
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
# Given that the files containing the training data have 100% chance of containing data from their respective classes
# Prior is 1
total_samples = len(train_x) + len(train_y)
print("Total training samples: {}".format(total_samples))

prior_x = 0.9
prior_y = 0.1

# This is the current dataset to be tested.
testing_dataset = test_xy_126
validation_dataset = test_xy_126_label

likelihood_x = sh.likelihood(testing_dataset, train_x_mean, train_x_covariance)
likelihood_y = sh.likelihood(testing_dataset, train_y_mean, train_y_covariance)

# Compute posteriors from likelihood and prior

# Compute the evidence P(xy)
evidence_xy = likelihood_x * prior_x+likelihood_y * prior_y; # Probability of x= probability that is in class 1 and the probability of class 1 

posterior_x= (likelihood_x*prior_x)/evidence_xy
posterior_y= (likelihood_y*prior_y)/evidence_xy

# predicting each element's class assignment and noting it down in a list.
predictedClass = []
for i in range(len(testing_dataset)):
	if (posterior_x[i] > posterior_y[i]):
		predictedClass.append(1)
	else:
		predictedClass.append(2)

# Generating confusion matrix
correctlyClassified = 0
incorrecltyClassified = 0

# todo: Currently this matrix is coupled, we need to assess individually the models for each class
for i in range(len(validation_dataset)):
	if (predictedClass[i] == validation_dataset[i]):
		correctlyClassified += 1
	else:
		incorrecltyClassified += 1

print("True Positives: {}".format(correctlyClassified))
print("False Positives: {}".format(incorrecltyClassified))
print("Accuracy: {}".format(float(correctlyClassified) / len(validation_dataset)))
print("ErrorRate: {}".format(float(incorrecltyClassified) / len(validation_dataset)))

