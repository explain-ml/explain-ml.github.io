# Bayesian model selection

Author: [Zeel B Patel](https://patel-zeel.github.io/)

http://mlg.eng.cam.ac.uk/teaching/4f13/1920/bayesian%20finite%20regression.pdf (Last slide)
http://mlg.eng.cam.ac.uk/teaching/4f13/1920/marginal%20likelihood.pdf

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', size=14)
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from time import time

### Pseudo-random data (degree 3 polynomial)

np.random.seed(0)

N = 100
sigma_n = 5 # noise in data
sigma_w = 100 # parameter variance

x = np.linspace(-2,2,N).reshape(-1,1)
f_x = 4*x**3 + 3*x**2 + 2*x + 1
epsilon = np.random.normal(loc=0, scale=sigma_n, size=N).reshape(-1,1)
y = f_x + epsilon

plt.scatter(x, y);
plt.xlabel('x');plt.ylabel('y');

### Marginal likelihood pdf

def LogMarginalLikelihoodPdf(x, y):
    return np.log(scipy.stats.multivariate_normal.pdf(y.squeeze(), np.zeros(N), (x@x.T)*sigma_w**2 + np.eye(N)*sigma_n**2))

### Models

x_M0 = x.copy()
x_M1 = np.hstack([np.ones((N,1)), x])
x_M2 = np.hstack([np.ones((N,1)), x, x**2])
x_M3 = np.hstack([np.ones((N,1)), x, x**2, x**3])
x_M4 = np.hstack([np.ones((N,1)), x, x**2, x**3, x**4])
x_M5 = np.hstack([np.ones((N,1)), x, x**2, x**3, x**4, x**5])
x_M6 = np.hstack([np.ones((N,1)), x, x**2, x**3, x**4, x**5, x**6])
x_M7 = np.hstack([np.ones((N,1)), x, x**2, x**3, x**4, x**5, x**6, x**7])

### Model selection

scores = [LogMarginalLikelihoodPdf(x_M, y) for x_M in [x_M0, x_M1, x_M2, x_M3, x_M4, x_M5, x_M6, x_M7]]
plt.plot(scores);
plt.xlabel('Degree of polynomial');
plt.ylabel('Log Marginal Likelihood \n(Higher is better)');
plt.xticks(range(len(scores)));

We can infer that Degree 3 polynomial is best suited to model current data.

### Model selection with parameter optimization

def NegLogMarginalLikelihoodPdf(params, x, y): # Negative log marginal likelihood (written in GP fashion)
    sigma_n, sigma_w = params
    K = (x@x.T)*sigma_w**2 + np.eye(N)*sigma_n**2
    K_inv = np.linalg.pinv(K)
    nll = 0.5*y.T@K_inv@y + 0.5*np.log(np.linalg.det(K)) + (len(y)/2)*np.log(2*np.pi)
    return nll[0,0]

Negscores = []
sig_n_list = []
sig_w_list = []
sig_w = 10
sig_n = 10
for i, x_M in enumerate([x_M0, x_M1, x_M2, x_M3, x_M4, x_M5, x_M6, x_M7]):
    init = time()
    result = minimize(fun=NegLogMarginalLikelihoodPdf, x0=(sig_n, sig_w), args=(x_M, y))
    sig_n_opt, sig_w_opt = result.x
    sig_n_list.append(sig_n_opt)
    sig_w_list.append(sig_w_opt)
    print(f'degree = {i}, sigma_w={sig_w_opt}, sig_n_opt={sig_n_opt}', 'time:',time()-init,'seconds')
    Negscores.append(NegLogMarginalLikelihoodPdf((sig_n_opt, sig_w_opt), x_M, y))
plt.plot(Negscores);
plt.xlabel('Degree of polynomial');
plt.ylabel('Neg Log Marginal Likelihood \n(Lower is better)');
plt.xticks(range(len(Negscores)));

We can see the best set of hyper-parameters selected by gradient descent on negative log likelihood for indivisual models (degree of polynomial).

## Drawbacks

1. This method does not scale well with big data (while doing parameter optimization), because of matrix inversion.
    * Potential solution: optimize parameters with chepaer methods, then use Marginal likelihood for model selection

## Using empirical bayes (Bishop)

http://krasserm.github.io/2019/02/23/bayesian-linear-regression/

alpha = 1/sigma_w**2
beta = 1/sigma_n**2

### Marginal likelihood pdf

def posterior(Phi, t, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N

def log_marginal_likelihood(Phi, t, alpha, beta):
    """Computes the log of the marginal likelihood."""
    N, M = Phi.shape

    m_N, _, S_N_inv = posterior(Phi, t, alpha, beta, return_inverse=True)
    
    E_D = beta * np.sum((t - Phi.dot(m_N)) ** 2)
    E_W = alpha * np.sum(m_N ** 2)
    
    score = M * np.log(alpha) + \
            N * np.log(beta) - \
            E_D - \
            E_W - \
            np.log(np.linalg.det(S_N_inv)) - \
            N * np.log(2 * np.pi)

    return 0.5 * score

Bscores = [log_marginal_likelihood(x_M, y, alpha, beta) for x_M in [x_M0, x_M1, x_M2, x_M3, x_M4, x_M5, x_M6, x_M7]]
plt.plot(Bscores);
plt.xlabel('Degree of polynomial');
plt.ylabel('Log Marginal Likelihood \n(Higher is better)');
plt.xticks(range(len(Bscores)));

Results are exactly the same as the previous method.

### Paramater tuning

def fit(Phi, t, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5, verbose=False):
    """
    Jointly infers the posterior sufficient statistics and optimal values 
    for alpha and beta by maximizing the log marginal likelihood.
    
    Args:
        Phi: Design matrix (N x M).
        t: Target value array (N x 1).
        alpha_0: Initial value for alpha.
        beta_0: Initial value for beta.
        max_iter: Maximum number of iterations.
        rtol: Convergence criterion.
        
    Returns:
        alpha, beta, posterior mean, posterior covariance.
    """
    
    N, M = Phi.shape

    eigenvalues_0 = np.linalg.eigvalsh(Phi.T.dot(Phi))

    beta = beta_0
    alpha = alpha_0

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha

        eigenvalues = eigenvalues_0 * beta

        m_N, S_N, S_N_inv = posterior(Phi, t, alpha, beta, return_inverse=True)

        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        alpha = gamma / np.sum(m_N ** 2)

        beta_inv = 1 / (N - gamma) * np.sum((t - Phi.dot(m_N)) ** 2)
        beta = 1 / beta_inv

        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(beta_prev, beta, rtol=rtol):
            if verbose:
                print(f'Convergence after {i + 1} iterations.')
            return alpha, beta, m_N, S_N

    if verbose:
        print(f'Stopped after {max_iter} iterations.')
    return alpha, beta, m_N, S_N

for i, x_M in enumerate([x_M0, x_M1, x_M2, x_M3, x_M4, x_M5, x_M6, x_M7]):
    init = time()
    alpha, beta, m_N, S_N = fit(x_M, y, rtol=1e-5, verbose=True)
    print('Degree',i,'sigma_w',1/np.sqrt(alpha),'sigma_n', 1/np.sqrt(beta))
    print('Degree',i,'sigma_w',sig_w_list[i],'sigma_n', sig_n_list[i], '(Earlier method)')
    print('time:',time()-init,'seconds')

This method is extremely fast than the previous method. We get close answers while using any of the two methods.