http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html

https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php

Coursera course on ML by UW (Emily Fox)

Expectation Maximisation Based GMM for 1d data

Basic Algorithm

- Assume K clusters
- Prior of sample being in that cluster

$$\begin{array}{l}
0 \leq \pi_{k} \leq 1 \\
\sum_{k=1}^{K} \pi_{k}=1
\end{array}
$$

- Cluster assignment ($z$)for $i^{th}$ observation to $k^{th}$ cluster:
$p\left(z_{i}=k\right)=\pi_{k} $

- Given observation $\left(x_{i}\right)$ s from cluster $k$, what's the ikelihood of seeing $\mathbf{x}_{i} ?$ 

$p\left(x_{i} \mid z_{i}=k, \mu_{k}, \Sigma_{k}\right)=N\left(x_{i} \mid \mu_{k}, \Sigma_{k}\right)$

- Posterior that a sample $x_i$ is sampled from $k^{th}$ cluster is:

$$ p(z_i=k|x_i, \mu_{k}, \Sigma_{k}, \pi_k) = \frac{\pi_k \times N\left(x_{i} \mid \mu_{k}, \Sigma_{k}\right)}{\sum_{k=1}^{K}\pi_k \times N\left(x_{i} \mid \mu_{k}, \Sigma_{k}\right)} $$





import mediapy
mediapy.show_image(mediapy.read_image("../figs/gmm-responsibility.png"))

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)


x1 = np.random.normal(0, 1, 20)

x2 = np.random.normal(4, 1, 10)


x = np.hstack((x1, x2))
c_true = np.hstack((np.zeros(len(x1)), np.ones(len(x2))))

plt.scatter(x, np.ones_like(x), c=c_true)

n_points = len(x)
n_clusters = 2

responsibility = np.abs(np.random.randn(n_points, n_clusters))
responsibility = responsibility/responsibility.sum(axis=1).reshape(-1, 1)
responsibility

# Step 1

Randomly assign points to cluster

## Get MLE estimate





means = {}
variances = {}
pis = {}
num_points = {}

for cl in range(n_clusters):
    num_points[cl] = np.sum(responsibility[:, cl])
    means[cl] = np.sum(responsibility[:, cl]*x)/num_points[cl]
    variances[cl] =(responsibility[:, cl]*(x-means[cl]))@(x-means[cl])/num_points[cl]
    pis[cl] = num_points[cl]/len(x)*1.

means

variances

pis

## Getting state distribution

from scipy.stats import norm





for i, point in enumerate(x):
    for cl in range(n_clusters):
        responsibility[i, cl] = norm.pdf(point, means[cl], np.sqrt(variances[cl]))*pis[cl]
    responsibility[i] = responsibility[i]/np.sum(responsibility[i])
        


responsibility



norm.pdf(x[0], means[cl], np.sqrt(variances[cl]))*pis[cl], norm.pdf(x[1], means[cl], np.sqrt(variances[cl]))*pis[cl]

import os

Total

x_pl = np.linspace(-5, 15, 1000)
for iteration in range(50):

    means = {}
    variances = {}
    pis = {}
    num_points = {}

    for cl in range(n_clusters):
        num_points[cl] = np.sum(responsibility[:, cl])
        means[cl] = np.sum(responsibility[:, cl]*x)/num_points[cl]
        variances[cl] =(responsibility[:, cl]*(x-means[cl]))@(x-means[cl])/num_points[cl]
        pis[cl] = num_points[cl]/len(x)*1.
    print(means)
    norm_obj = {}
    for cl in range(n_clusters):
        norm_obj[cl] = norm(loc=means[cl], scale=np.sqrt(variances[cl]))
        plt.plot(x_pl, norm_obj[cl].pdf(x_pl),
                lw=3, alpha=0.6, label='Cluster {}'.format(cl))
    plt.legend()
    plt.scatter(x, np.zeros_like(x), c='k', s=300, marker='|')
    plt.title("Iteration: {}".format(str(iteration).zfill(2)))
    plt.savefig(os.path.expanduser("~/git/explain-ml/figs/gmm/{}.jpg".format(str(iteration).zfill(2))))
    
    plt.clf()
    for i, point in enumerate(x):
        for cl in range(n_clusters):
            responsibility[i, cl] = norm.pdf(point, means[cl], np.sqrt(variances[cl]))*pis[cl]
        responsibility[i] = responsibility[i]/np.sum(responsibility[i])

    


!convert -delay 20 -loop 0 ../figs/gmm/*.jpg ../figs/gmm/gmm.gif

![](/Users/nipun/git/explain-ml/figs/gmm.gif)

    !pip install -q mediapy


import mediapy as media

media.show_video(media.read_video("../figs/gmm.gif"), fps=2)

from scipy.stats import multivariate_normal
class GMM:
    def __init__(self, n_clusters=2, random_seed=0):
        self.n_clusters = n_clusters
        np.random.seed(random_seed)
    
    def fit(self, X, n_iter=100):
        self.n_samples, self.dim = X.shape
        self.n_iter = n_iter
        self.X = X
        self.responsibility = np.abs(np.random.random(size=(self.n_samples, self.n_clusters)))
        self.responsibility = self.responsibility/self.responsibility.sum(axis=1).reshape(-1, 1)
        self.means = {}
        self.covariances = {}
        self.pis = {}
        self.num_points = {}
        
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(X)
        
        for cl in range(self.n_clusters):
            self.pis[cl] = np.random.random()
            #self.means[cl] = np.random.randn(self.dim)
            self.means[cl] = km.cluster_centers_[cl]
            temp = np.abs(np.random.randn(self.dim, self.dim))
            self.covariances[cl] = np.transpose(temp)@temp
            self.covariances[cl] = 0.5+np.random.uniform()*np.ones((self.dim, self.dim))
            self.covariances[cl] = self.covariances[cl] - 0.1*np.eye(self.dim, self.dim)
            self.covariances[cl] = self.covariances[cl].T@self.covariances[cl]
            self.covariances[cl] = temp.T@temp
            print(self.covariances[cl])
            #print(self.covariances[cl].shape)
        print(self.means)
        # Normalizing pi
        for cl in range(self.n_clusters):
            #self.pis[cl] = self.pis[cl]/np.sum(list(self.pis.values()))
            self.pis[cl] = 1.0/self.n_clusters
        for i in range(self.n_iter):
            #self.m_step()
            self.e_step()
            self.m_step()
            print("****"*20, "Iteration:", i)
            print(self.means)
            
    
    def e_step(self):
        #print("----"*10+"E-Step")
        self.norm_obj = {}
        for cl in range(self.n_clusters):
            self.norm_obj[cl] = multivariate_normal(mean=self.means[cl], cov=self.covariances[cl])
        
        for i, point in enumerate(self.X):
            for cl in range(n_clusters):
                self.responsibility[i, cl] = self.norm_obj[cl].pdf(point)*self.pis[cl]
        
            self.responsibility[i] = self.responsibility[i]/np.sum(self.responsibility[i])

        #print(self.responsibility)
        
        
        
    
    def m_step(self):
        #print("----"*10+"M-Step")
        for cl in range(self.n_clusters):
            #print("--"*20)
            self.num_points[cl] = np.sum(self.responsibility[:, cl])
            self.means[cl] = np.sum(self.responsibility[:, cl].reshape(-1, 1)*self.X, axis=0)/self.num_points[cl]
            self.covariances[cl] = 0.01*np.eye(self.dim, self.dim)+((g.X-g.means[0]).T* g.responsibility[:, 0])@(self.X-self.means[cl])/self.num_points[cl]
            self.pis[cl] = self.num_points[cl]/len(self.X)*1.
        print(self.covariances)
    



X, y = make_blobs(cluster_std=0.3)
g = GMM(n_clusters=3)

from sklearn.datasets import make_blobs



g.fit(X, n_iter=50, )

(g.X-g.means[0]).T.shape, g.responsibility[:, 0].shape

g.covariances[0]
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


is_pos_def(g.covariances[0])

plt.scatter(X[:, 0], X[:, 1],c=y)
for pos, covar in zip(list(g.means.values()), list(g.covariances.values())):
    print(pos)
    print(covar)
    draw_ellipse(pos, covar, alpha=0.2)




from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

from sklearn import mixture
model = mixture.GaussianMixture(n_components=3, covariance_type='full')

plot_gmm(model, X)

model.covariances_

model.means_

g.means

g.covariances

using Bayesian

Following: https://pyro.ai/examples/gmm.html

import seaborn as sns
import pymc3 as pm
import arviz as az
import theano.tensor as tt

import numpy as np
np.random.seed(0)


x1 = np.random.normal(0, 1, 20)

x2 = np.random.normal(4, 1, 10)


x_gmm = np.hstack((x1, x2))
c_true = np.hstack((np.zeros(len(x1)), np.ones(len(x2))))

ndata = len(x_gmm)
basic_model = pm.Model()


with basic_model:

    # Priors for unknown model parameters
    weights = pm.Dirichlet("pi", [20, 10])
    scale = pm.Lognormal("scale", 0, 3, shape=2)
    locs = pm.Normal("locations", -5, 5, shape=2)
    category = pm.Categorical("category", p=weights, shape=ndata)
    x = pm.Data('input', x_gmm)
    points = pm.Normal("obs", mu=locs[category], sigma=scale[category], observed=x_gmm)

pm.model_to_graphviz(basic_model.model)



map_estimate = pm.find_MAP(model=basic_model)

map_estimate

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(2000,return_inferencedata=False,tune=1000)

