import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import seaborn as sns
%matplotlib inline

https://www.youtube.com/watch?v=kYWHfgkRc9s

rv = expon()

Plotting pdf of exponential distribution

x = np.linspace(0, 10, 1000)
plt.plot(x, rv.pdf(x))

Generating samples from uniform distribution

np.random.uniform(0, 10)

sns.kdeplot(np.random.uniform(0, 10, 100))

sns.kdeplot(np.random.uniform(0, 10, 1000000))

x = np.linspace(0, 10, 1000)
plt.plot(x, rv.pdf(x),'k',lw=2)

samples_uniform_x = np.random.uniform(0, 10, 100000)
samples_uniform_y = np.random.uniform(0, 1, 100000)


pdfs = rv.pdf(samples_uniform_x)

idx = samples_uniform_y < pdfs

plt.scatter(samples_uniform_x[idx], samples_uniform_y[idx],alpha=0.3, color='green',s=0.1,label="Accepted")
plt.scatter(samples_uniform_x[~idx], samples_uniform_y[~idx],alpha=0.3, color='red',s=0.1,label="Rejected")
plt.legend()


plt.hist(samples_uniform_x[idx], bins=100);

def rejection_sampling(pdf, lower_support, upper_support, samples=1000, y_max = 1):
    #x = np.linspace(0, 10, 1000)
    #plt.plot(x, pdf(x),'k',lw=2)

    samples_uniform_x = np.random.uniform(lower_support, upper_support, samples)
    samples_uniform_y = np.random.uniform(0, y_max, samples)


    pdfs = pdf(samples_uniform_x)

    idx = samples_uniform_y < pdfs

    plt.scatter(samples_uniform_x[idx], samples_uniform_y[idx],alpha=0.6, color='green',s=0.1,label="Accepted")
    plt.scatter(samples_uniform_x[~idx], samples_uniform_y[~idx],alpha=0.6, color='red',s=0.1,label="Rejected")
    plt.title(samples_uniform_x[idx].mean())
    plt.legend()



from scipy.stats import norm
scale =1
rv = norm(loc=0, scale=scale)
pdf = rv.pdf
rejection_sampling(pdf, -5, 5, 10000)
x = np.linspace(-5, 5, 1000)
plt.plot(x, pdf(x),'k',lw=2)

from scipy.stats import norm
scale =0.1
rv = norm(loc=0, scale=scale)
pdf = rv.pdf
rejection_sampling(pdf, -5, 5, 10000)
x = np.linspace(-5, 5, 1000)
plt.plot(x, pdf(x),'k',lw=2)

scale =0.1
rv = norm(loc=1, scale=scale)
pdf = rv.pdf
rejection_sampling(pdf, -5, 5, 50000,y_max=(1/scale)/(np.sqrt(2*np.pi)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, pdf(x),'k',lw=2)

from scipy.stats import gamma

rv = gamma(1)
pdf = rv.pdf
rejection_sampling(pdf, 0, 5, 10000)
x = np.linspace(0, 5, 1000)
plt.plot(x, pdf(x),'k',lw=2)

rv = gamma(2)
pdf = rv.pdf
rejection_sampling(pdf, 0, 5, 10000)
x = np.linspace(0, 5, 1000)
plt.plot(x, pdf(x),'k',lw=2)

rv = gamma(10)
pdf = rv.pdf
rejection_sampling(pdf, 0, 10, 10000)
x = np.linspace(0, 10, 1000)
plt.plot(x, pdf(x),'k',lw=2)

