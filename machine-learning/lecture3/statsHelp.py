# File containing all the statistics calculations needed in one place.
# so I don't have to look at them any more than I need to.

import numpy as np
import matplotlib.pyplot as plt
import plottingFunctions as pf

def getMean(dataset):
    return np.mean(dataset)

def getCovariance(dataset):
    return np.cov(dataset)

def getPrior(datasetLabels):
    prior = []
    sampleCount = len(datasetLabels)

    for label in set(datasetLabels):
        count = datasetLabels.count(label)
        prior[label] = count / sampleCount

    return prior
