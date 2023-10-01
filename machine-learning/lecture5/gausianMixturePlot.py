from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

data_path = "2D568class.mat"
data = loadmat(data_path)
train5 = data["trn5_2dim"]/255
train6 = data["trn6_2dim"]/255
train8 = data["trn8_2dim"]/255

trainset = np.concatenate([train5, train6, train8])
np.random.seed(0)
np.random.shuffle(trainset)

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# concatenate the two datasets into the final training set
X_train = np.vstack([trainset])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type="full")
clf.fit(X_train)
labelsGMM = clf.predict(X_train)
# display predicted scores by the model as a contour plot
x = np.linspace(-10.0, 15.0)
y = np.linspace(-10.0, 15.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
)
CB = plt.colorbar(CS, shrink=0.8, extend="both")
plt.scatter(X_train[:, 0], X_train[:, 1], 0.8, labelsGMM)

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.savefig("gausianMixPlot.png", format="png")