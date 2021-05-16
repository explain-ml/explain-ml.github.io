# GP v/s Deep GP on 2d data

Author: [Nipun Batra](https://nipunbatra.github.io/), [Zeel B Patel](https://patel-zeel.github.io/)

In this notebook, we will be comparing three models (all RBF kernel) on a simulated 2d data:
    
- GP
- GP with ARD
- Deep GP 

import GPy

import numpy as np
import pyDOE2
from sklearn.metrics import mean_squared_error
# Plotting tools
from matplotlib import pyplot as plt
from matplotlib import rc

rc('font', size=16)
rc('text', usetex=True)

import warnings
warnings.filterwarnings('ignore')

# GPy: Gaussian processes library
import GPy

We generate pseudo data using the following function,

$$
y = \frac{(x_1+2)(2-x_2^2)}{3}
$$

np.random.seed(0)
X = pyDOE2.doe_lhs.lhs(2, 9, random_state=0)
func = lambda x: ((x[:, 0]+2)*(2-x[:, 1])**2)/3
y = func(X)

plt.scatter(X[:,0], X[:, 1],s=y*50);

## GP without ARD

k_2d = GPy.kern.RBF(input_dim=2, lengthscale=1)

m = GPy.models.GPRegression(X, y.reshape(-1, 1), k_2d)
m.optimize();

m

x_1 = np.linspace(0, 1, 40)
x_2 = np.linspace(0, 1, 40)

X1, X2 = np.meshgrid(x_1, x_2)
X_new = np.array([(x1, x2) for x1, x2 in zip(X1.ravel(), X2.ravel())])
Y_true = func(X_new)

Y_pred, Y_cov = m.predict(X_new)
Y_95 = 2*np.sqrt(Y_cov)

fig, ax = plt.subplots(1,2,figsize=(12,4))
mp = ax[0].contourf(X1, X2, Y_pred.reshape(*X1.shape), levels=30)
fig.colorbar(mp, ax=ax[0])
ax[0].set_title(f'Predictive mean, RMSE = {mean_squared_error(Y_true, Y_pred, squared=False).round(7)}');

mp = ax[1].contourf(X1, X2, Y_95.reshape(*X1.shape), levels=30)
ax[1].set_title(f'Predictive variance (95\% confidence)')
fig.colorbar(mp, ax=ax[1]);

## GP with ARD

k_2d_ARD = GPy.kern.RBF(input_dim=2, lengthscale=1, ARD=True)

m = GPy.models.GPRegression(X, y.reshape(-1, 1), k_2d_ARD)
m.optimize();

m

Y_pred, Y_cov = m.predict(X_new)
Y_95 = 2*np.sqrt(Y_cov)

fig, ax = plt.subplots(1,2,figsize=(12,4))
mp = ax[0].contourf(X1, X2, Y_pred.reshape(*X1.shape), levels=30)
fig.colorbar(mp, ax=ax[0])
ax[0].set_title(f'Predictive mean, RMSE = {mean_squared_error(Y_true, Y_pred, squared=False).round(7)}');

mp = ax[1].contourf(X1, X2, Y_95.reshape(*X1.shape), levels=30)
ax[1].set_title(f'Predictive variance (95\% confidence)')
fig.colorbar(mp, ax=ax[1]);

m.parameters[0].lengthscale

We see that lengthscale for $x_1$ is higher because of linear relationship with the output and lengthscale for $x_2$ is higher due to quadratic relationship with the output.

## Deep GP

import deepgp
layers = [1, 1,  X.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []
for i in layers[1:]:
    kernels += [GPy.kern.RBF(i, ARD=True)]
    
m = deepgp.models.DeepGP(layers,Y=y.reshape(-1, 1), X=X, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=4, back_constraint=False)

m.optimize()

m

Y_pred, Y_cov = m.predict(X_new)
Y_95 = 2*np.sqrt(Y_cov)

fig, ax = plt.subplots(1,2,figsize=(12,4))
mp = ax[0].contourf(X1, X2, Y_pred.reshape(*X1.shape), levels=30)
fig.colorbar(mp, ax=ax[0])
ax[0].set_title(f'Predictive mean, RMSE = {mean_squared_error(Y_true, Y_pred, squared=False).round(3)}');

mp = ax[1].contourf(X1, X2, Y_95.reshape(*X1.shape), levels=30)
ax[1].set_title(f'Predictive variance (95\% confidence)')
fig.colorbar(mp, ax=ax[1]);

Not working
Pinball_plot : https://github.com/lawrennd/talks/blob/gh-pages/deepgp_tutorial.py