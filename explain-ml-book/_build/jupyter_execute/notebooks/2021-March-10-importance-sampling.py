https://www.youtube.com/watch?v=TNZk8lo4e-Q&t=2733s

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy.stats import norm

# Likelihood
def l(theta):
    return 4+ np.sin(theta) - (theta**2)/3

x = np.linspace(-3, 3, 1000)

prior = norm(0).pdf

plt.plot(x, prior(x),label='Prior')
plt.plot(x, l(x),label='Likelihood')
plt.legend()

p(100)

q_rvs = norm(loc=0, scale=10)
q = q_rvs.pdf

plt.plot(x, q(x))

sns.kdeplot(q_rvs.rvs(size=20000))

q_samples = q_rvs.rvs(size=10000)

prior_eval_q = prior(q_samples)

likelihood_eval_q = l(q_samples)

z = (prior_eval_q*likelihood_eval_q/q_samples).mean()

z

Importance sampling for linear regression simplified setting

$y \sim \mathcal{N}(\theta x, \sigma^2)$ 

theta_gt = 4
sigma_gt = 1


# generate some samples

x = np.random.uniform(0, 1, size = 100)
y = np.random.normal(theta_gt*x, sigma_gt)


plt.scatter(x, y)

# Proposal

q_rvs = norm(loc=3, scale=10)
q = q_rvs.pdf

q

prior = norm(0).pdf



prior

# Likelihood
import scipy.stats
def l(theta):
    
    return np.prod(scipy.stats.norm(theta*x, 3).pdf(y))

xu = np.linspace(2, 6, 1000)
k = []
for xt in xu:
    k.append(l(xt))
plt.plot(xu, k)


plt.hist(scipy.stats.norm(10*x, 1).pdf(y), density=False, bins=20)

n_samples = 1000
q_samples = q_rvs.rvs(size=n_samples)
#weights = np.multiply(p_likelihood,  p_prior) / p_proposal


plt.hist(q_samples)

w = np.zeros(n_samples)
for i in range(n_samples):
    theta_i = q_samples[i]
    likelihood_i = l(theta_i)
    prior_i = prior(theta_i)
    q_i = q_rvs.pdf(theta_i)
    w_i = likelihood_i*prior_i/q_i
    w[i] = w_i

np.mean(w)

post = np.zeros(n_samples)
for i in range(n_samples):
    theta_i = q_samples[i]
    likelihood_i = l(theta_i)
    prior_i = prior(theta_i)
    post_i = likelihood_i*prior_i
    post[i] = post_i/np.mean(w)

plt.hist(post)

