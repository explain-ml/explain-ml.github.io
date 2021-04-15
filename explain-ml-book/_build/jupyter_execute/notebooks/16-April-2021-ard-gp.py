# Automatic relevance determination (ARD)

import scipy.stats
import GPy
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

rc('text', usetex=True)
rc('font', size=16)

To understand the concept of ARD, let us generate a symthetic dataset where all features are not equally important.

np.random.seed(0)
N = 400
X = np.empty((N, 3))
y = np.empty((N, 1))

cov = [[1,0,0,0.99],[0,1,0,0.6],[0,0,1,0.1],[0.99,0.6,0.1,1]]

samples = np.random.multivariate_normal(np.zeros(4), cov, size=N)

X[:,:] = samples[:,:3]
y[:,:] = samples[:,3:4]
print('Correlation between X1 and y', np.corrcoef(X[:,0], y.ravel())[1,0])
print('Correlation between X2 and y', np.corrcoef(X[:,1], y.ravel())[1,0])
print('Correlation between X3 and y', np.corrcoef(X[:,2], y.ravel())[1,0])

Let us fit a GP model with a common lengthscale for all features.

model = GPy.models.GPRegression(X, y, GPy.kern.RBF(input_dim=3, ARD=False))
model.optimize()
model

Visualizing fit over $X_1$

model.plot(visible_dims=[0]);

Visualizing fit over $X_3$

model.plot(visible_dims=[2]);

Now, let us turn on the ARD and see the values of lengthscales learnt.

model = GPy.models.GPRegression(X, y, GPy.kern.RBF(input_dim=3, ARD=True))
model.optimize()
model.kern.lengthscale

We can see that the lengthscale for $X_3$ is abnormally larger than the other two due to lowest correlation with data.

## Real-data

Let us try a real dataset and see what insights we can get by ARD experiment on it.

from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
y = y.reshape(-1,1)
X.shape, y.shape

Let us see what do we get from ARD enabled GP fit.

model = GPy.models.GPRegression(X, y, GPy.kern.RBF(input_dim=13, ARD=True))
model.optimize()
model.kern.lengthscale

We can see some features seem more important (e.g. `[5],[7]`) and others do not. Let us verify this visually.

plt.scatter(X[:,5], y);

plt.scatter(X[:,1], y);

We can see a strong patern in `[5]` but we can not see any patterns in `[1]`.