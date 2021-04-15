import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
import seaborn as sns
import pymc3 as pm
import arviz as az

%matplotlib inline

np.random.seed(0)
X, y = make_moons(n_samples=200, noise=0.1)

plt.scatter(X[:, 0], X[:, 1],c=y)
X_concat = np.hstack((np.ones((len(y), 1)), X))
X_concat.shape

basic_model = pm.Model()
shapes = [4, 5, 1]

with basic_model:
    w1 = pm.Normal("w1", mu=1, sigma=10, shape=(X_concat.shape[1], shapes[0]))
    w2 = pm.Normal("w2", mu=2, sigma=10, shape=(shapes[0], shapes[1]))
    w3 = pm.Normal("w3", mu=3, sigma=10, shape=(shapes[1], shapes[2]))
                   
    X_ = pm.Data('features', X_concat)
    y_ = pm.Data('targets', y)
    # Expected value of outcome
    
    a_1 = pm.math.sigmoid(pm.math.dot(X_, w1))
    a_2 = pm.math.sigmoid(pm.math.dot(a_1, w2))
    a_3 = pm.math.sigmoid(pm.math.dot(a_2, w3))
    
    
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli("Y_obs", p=a_3, observed=y_)

pm.model_to_graphviz(basic_model.model)

map_estimate = pm.find_MAP(model=basic_model)
map_estimate

with basic_model:
    inference = pm.ADVI()
    approx = pm.fit(n=300, method=inference)

trace = approx.sample(draws=50)

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(20,return_inferencedata=False,tune=10)

az.plot_trace(trace);

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

X_test = np.c_[xx.ravel(), yy.ravel()]
X_test_concat = np.hstack((np.ones((len(X_test), 1)), X_test))
X_test_concat.shape

with basic_model:
    pm.set_data({'features': X_test_concat})
    posterior = pm.sample_posterior_predictive(trace)

Z = posterior['Y_obs']
posterior['Y_obs'].shape, X_test_concat.shape

for i in range(len(Z))[:500]:
    plt.contour(xx, yy, Z[i].reshape(xx.shape), alpha=0.01)
plt.scatter(X[:, 0], X[:, 1],c=y, zorder=10)

FOllowing code inspired from": https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html

pred = posterior['Y_obs'].mean(axis=0)>0.5
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
fig, ax = plt.subplots(figsize=(8, 4))
contour = plt.contourf(xx, yy, posterior['Y_obs'].mean(axis=0).reshape(xx.shape),cmap=cmap)
#ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
#ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
cbar = plt.colorbar(contour, ax=ax)
#_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
#cbar.ax.set_ylabel("Posterior predictive mean probability of class label = 0");


pred = posterior['Y_obs'].mean(axis=0)>0.5
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
fig, ax = plt.subplots(figsize=(8, 4))
contour = plt.contourf(xx, yy, posterior['Y_obs'].std(axis=0).reshape(xx.shape),cmap=cmap)
#ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
#ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
cbar = plt.colorbar(contour, ax=ax)
#_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
#cbar.ax.set_ylabel("Posterior predictive mean probability of class label = 0");

