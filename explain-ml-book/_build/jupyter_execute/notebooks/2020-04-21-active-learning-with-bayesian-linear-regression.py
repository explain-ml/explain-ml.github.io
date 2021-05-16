# Active Learning with Bayesian Linear Regression

> A programming introduction to Active Learning with Bayesian Linear Regression.

Author: [Zeel B Patel](https://patel-zeel.github.io/), [Nipun Batra](https://nipunbatra.github.io/)

### A quick wrap-up for Bayesian Linear Regression (BLR)

We have a feature matrix $X$ and a target vector $\mathbf{y}$. We want to obtain $\boldsymbol{\theta}$ vector in such a way that the error $\boldsymbol{\epsilon}$ for the following equation is minimum.

\begin{equation}
Y = X\boldsymbol{\theta} + \boldsymbol{\epsilon}
\end{equation}
Prior PDF for $\boldsymbol{\theta}$ is,

\begin{equation}
p(\boldsymbol{\theta}) \sim \mathcal{N}(\mathbf{m}_0, S_0)
\end{equation}

Where $S_0$ is prior covariance matrix, and $\mathbf{m}_0$ is prior mean. 

Posterier PDF can be given as,

\begin{equation}
p(\boldsymbol{\theta}|X,\mathbf{y}) \sim \mathcal{N}(\boldsymbol{\theta} | \mathbf{m}_n, S_n) \\
S_n = (S_0^{-1} + \sigma_{mle}^{-2}X^TX) \\
\mathbf{m}_n = S_n(S_0^{-1}\mathbf{m}_0+\sigma_{mle}^{-2}X^T\mathbf{y})
\end{equation}

Maximum likelihood estimation of $\sigma$ can be calculated as,  

\begin{equation}
\theta_{mle} = (X^TX)^{-1}X^T\mathbf{y} \\
\sigma_{mle} = ||\mathbf{y} - X\boldsymbol{\theta}_{mle}||
\end{equation}

Finally, predicted mean $\hat{\mathbf{y}}_{mean}$ and predicted covariance matrix $\hat{Y}_{cov}$ can be given as,

\begin{equation}
\hat{\mathbf{y}} \sim \mathcal{N}(\hat{\mathbf{y}}_{mean}, \hat{\mathbf{y}}_{cov}) \\
\hat{\mathbf{y}}_{mean} = X\mathbf{m}_n \\
\hat{Y}_{cov} = XS_nX^T
\end{equation}

Now, let's put everything together and write a class for [Bayesian Linear Regression](https://nipunbatra.github.io/blog/ml/2020/02/20/bayesian-linear-regression.html).

### Creating scikit-learn like class with fit predict methods for BLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import warnings
warnings.filterwarnings('ignore')
seed = 0 # random seed for train_test_split

rc('font', size=16)
rc('text', usetex=True)

class BLR():
  def __init__(self,S0, M0): # M0 -> prior mean, S0 -> prior covariance matrix
    self.S0 = S0
    self.M0 = M0

  def fit(self,x,y, return_self = False):
    self.x = x
    self.y = y

    # Maximum likelihood estimation for sigma parameter
    theta_mle = np.linalg.pinv(x.T@x)@(x.T@y)
    sigma_2_mle = np.linalg.norm(y - x@theta_mle)**2
    sigma_mle = np.sqrt(sigma_2_mle)

    # Calculating predicted mean and covariance matrix for theta
    self.SN = np.linalg.pinv(np.linalg.pinv(self.S0) + (sigma_mle**-2)*x.T@x)
    self.MN = self.SN@(np.linalg.pinv(self.S0)@self.M0 + (sigma_mle**-2)*(x.T@y).squeeze())

    # Calculating predicted mean and covariance matrix for data
    self.pred_var = x@self.SN@x.T
    self.y_hat_map = x@self.MN
    if return_self:
      return (self.y_hat_map, self.pred_var)
    
  def predict(self, x):
    self.pred_var = x@self.SN@x.T
    self.y_hat_map = x@self.MN
    return (self.y_hat_map, self.pred_var)

  def plot(self, s=1): # s -> size of dots for scatter plot
    individual_var = self.pred_var.diagonal()
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(self.x[:,1], self.y_hat_map, color='black', label='model')
    plt.fill_between(self.x[:,1], self.y_hat_map-individual_var, self.y_hat_map+individual_var, alpha=0.4, color='black', label='uncertainty')
    plt.scatter(self.x[:,1], self.y, label='actual data',s=s)
    plt.title('MAE is '+str(np.mean(np.abs(self.y - self.y_hat_map))))
    plt.legend()

### Creating & visualizing dataset

To start with, let's create a random dataset with degree 3 polynomial function with some added noise.

\begin{equation}
Y = (5X^3 - 4X^2 + 3X - 2) + \mathcal{N}(0,1)
\end{equation}

np.random.seed(seed)
X_init = np.linspace(-1, 1, 1000)
noise = np.random.randn(1000, )
Y = (5 * X_init**3 - 4 * X_init**2 + 3 * X_init - 2) + noise

We'll try to fit a degree 5 polynomial function to our data.

X = PolynomialFeatures(degree=5, include_bias=True).fit_transform(X_init.reshape(-1,1))
N_features = X.shape[1]

plt.scatter(X[:,1], Y, s=0.5, label = 'data points')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

### Learning a BLR model on the entire data

We'll take $M_0$ (prior mean) as zero vector initially, assuming that we do not have any prior knowledge about $M_0$. We're taking $S_0$ (prior covariance) as the identity matrix, assuming that all coefficients are completely independent of each other.

S0 = np.eye(N_features)
M0 = np.zeros((N_features, ))
model = BLR(S0, M0)

model.fit(X, Y)

### Visualising the fit

model.plot(s=0.5)

This doesn't look like a good fit, right? Let's set the prior closer to the real values and visualize the fit again.

### Visualising the fit after changing the prior

np.random.seed(seed)
S0 = np.eye(N_features)
M0 = np.array([-2, 3, -4, 5, 0, 0]) + np.random.randn(N_features, )
model = BLR(S0, M0)

model.fit(X, Y)
model.plot(s=0.5)

Hmm, better. Now let's see how it fits after reducing the noise and setting the prior mean to zero vector again.

### Visualising the fit after reducing the noise

np.random.seed(seed)
X_init = np.linspace(-1, 1, 1000)
noise = np.random.randn(1000, ) * 0.5
Y = (5 * X_init**3 - 4 * X_init**2 + 3 * X_init - 2) + noise

S0 = np.eye(N_features)
M0 = np.zeros((N_features, ))
model = BLR(S0, M0)

model.fit(X, Y)
model.plot(s=0.5)

When the noise was high, the model tended to align with the prior. After keeping the prior closer to the original coefficients, the model was improved as expected. From the last plot, we can say that as noise reduces from the data, the impact of the prior reduces, and the model tries to fit the data more precisely. Therefore, we can say that when data is too noisy or insufficient, a wisely chosen prior can produce a precise fit.

### Intuition to Active Learning (Uncertainty Sampling) with an example

Let's take the case where we want to train a machine learning model to classify if a person is infected with COVID-19 or not, but the testing facilities for the same are not available so widely. We may have very few amounts of data for detected positive and detected negative patients. Now, we want our model to be highly confident or least uncertain about its results; otherwise, it may create havoc for wrongly classified patients, but, our bottleneck is labeled data. Thanks to active learning techniques, we can overcome this problem smartly. How?

We train our model with existing data and test it on all the suspected patients' data. Let's say we have an uncertainty measure or confidence level about each tested data point (distance from the decision boundary in case of SVM, variance in case of Gaussian processes, or Bayesian Linear Regression). We can choose a patient for which our model is least certain, and send him to COVID-19 testing facilities (assuming that we can send only one patient at a time). Now, we can include his data to the train set and test the model on everyone else. By following the same procedure repeatedly, we can increase the size of our train data and confidence of the model without sending everyone randomly for testing.

This method is called Uncertainty Sampling in Active Learning. Now let's formally define Active Learning. From Wikipedia,

*Active learning is a special case of machine learning in which a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs.*

Now, we'll go through the active learning procedure step by step.

### Train set, test set, and pool. What is what?

The train set includes labeled data points. The pool includes potential data points to query for a label, and the test set includes labeled data points to check the performance of our model. Here, we cannot actually do a query to anyone, so we assume that we do not have labels for the pool while training, and after each iteration, we include a data point from the pool set to the train set for which our model has the highest uncertainty.

So, the algorithm can be represented as the following,

1.  Train the model with the train set.
2.  Test the performance on the test set (This should keep improving).   
3.  Test the model with the pool.
4.  Query for the most uncertain datapoint from the pool.
5.  Add that datapoint into the train set.
6.  Repeat step 1 to step 5 for $K$ iterations ($K$ ranges from $0$ to the pool size).

### Creating initial train set, test set, and pool

Let's take half of the dataset as the test set, and from another half, we will start with some points as the train set and remaining as the pool. Let's start with 2 data points as the train set.

np.random.seed(seed)
X_init = np.linspace(-1, 1, 1000)
X = PolynomialFeatures(degree=5, include_bias=True).fit_transform(X_init.reshape(-1,1))
noise = np.random.randn(1000, ) * 0.5
Y = (5 * X_init**3 - 4 * X_init**2 + 3 * X_init - 2) + noise

train_pool_X, test_X, train_pool_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state=seed)
train_X, pool_X, train_Y, pool_Y = train_test_split(train_pool_X, train_pool_Y, train_size=2, random_state=seed)

Visualizing train, test and pool.

plt.scatter(test_X[:,1], test_Y, label='test set',color='r', s=2)
plt.scatter(train_X[:,1], train_Y, label='train set',marker='s',color='k', s=50)
plt.scatter(pool_X[:,1], pool_Y, label='pool',color='b', s=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

Let's initialize a few dictionaries to keep track of each iteration.

train_X_iter = {} # to store train points at each iteration
train_Y_iter = {} # to store corresponding labels to the train set at each iteration
models = {} # to store the models at each iteration
estimations = {} # to store the estimations on the test set at each iteration
test_mae_error = {} # to store MAE(Mean Absolute Error) at each iteration

### Training & testing initial learner on train set (Iteration 0)

Now we will train the model for the initial train set, which is iteration 0.

train_X_iter[0] = train_X
train_Y_iter[0] = train_Y

S0 = np.eye(N_features)
M0 = np.zeros((N_features, ))
models[0] = BLR(S0, M0)

models[0].fit(train_X_iter[0], train_Y_iter[0])

Creating a plot method to visualize train, test and pool with estimations and uncertainty.

def plot(ax, model, init_title=''):
  # Plotting the pool
  ax.scatter(pool_X[:,1], pool_Y, label='pool',s=1,color='r',alpha=0.4)
  
  # Plotting the test data
  ax.scatter(test_X[:,1], test_Y, label='test data',s=1, color='b', alpha=0.4)
  
  # Combining the test & the pool
  test_pool_X, test_pool_Y = np.append(test_X,pool_X, axis=0), np.append(test_Y,pool_Y)
  
  # Sorting test_pool for plotting
  sorted_inds = np.argsort(test_pool_X[:,1])
  test_pool_X, test_pool_Y = test_pool_X[sorted_inds], test_pool_Y[sorted_inds]
  
  # Plotting test_pool with uncertainty
  model.predict(test_pool_X)
  individual_var = model.pred_var.diagonal()
  ax.plot(test_pool_X[:,1], model.y_hat_map, color='black', label='model')
  ax.fill_between(test_pool_X[:,1], model.y_hat_map-individual_var, model.y_hat_map+individual_var
                  , alpha=0.2, color='black', label='uncertainty')
  
  # Plotting the train data
  ax.scatter(model.x[:,1], model.y,s=40, color='k', marker='s', label='train data')
  ax.scatter(model.x[-1,1], model.y[-1],s=80, color='r', marker='o', label='last added point')
  
  # Plotting MAE on the test set
  model.predict(test_X)
  ax.set_title(init_title+' MAE is '+str(np.mean(np.abs(test_Y - model.y_hat_map))))
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.legend(bbox_to_anchor=(1,1))

Plotting the estimations and uncertainty.

fig, ax = plt.subplots()
plot(ax, models[0])

Let's check the maximum uncertainty about any point for the model.

models[0].pred_var.diagonal().max()

Oops!! There is almost no uncertainty in the model. Why? let's try again with more train points.

train_pool_X, test_X, train_pool_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state=seed)
train_X, pool_X, train_Y, pool_Y = train_test_split(train_pool_X, train_pool_Y, train_size=7, random_state=seed)

train_X_iter[0] = train_X
train_Y_iter[0] = train_Y

S0 = np.eye(N_features)
M0 = np.zeros((N_features, ))
models[0] = BLR(S0, M0)

models[0].fit(train_X_iter[0], train_Y_iter[0])

fig, ax = plt.subplots()
plot(ax, models[0])

Now uncertainty is visible, and currently, it's high at the left-most points. We are trying to fit a degree 5 polynomial here. So our linear regression coefficients are 6, including the bias. If we choose train points equal to or lesser than 6, our model perfectly fits the train points and has no uncertainty. Choosing train points more than 6 induces uncertainty in the model.

Let's evaluate the performance on the test set.

estimations[0], _ = models[0].predict(test_X)
test_mae_error[0] = np.mean(np.abs(test_Y - estimations[0]))

Mean Absolute Error (MAE) on the test set is

test_mae_error[0]

### Moving the most uncertain point from the pool to the train set

In the previous plot, we saw that the model was least certain about the left-most point. We'll move that point from the pool to the train set and see the effect.

esimations_pool, _ = models[0].predict(pool_X)

Finding out a point having the most uncertainty.

in_var = models[0].pred_var.diagonal().argmax()
to_add_x = pool_X[in_var,:]
to_add_y = pool_Y[in_var]

Adding the point from the pool to the train set.

train_X_iter[1] = np.vstack([train_X_iter[0], to_add_x])
train_Y_iter[1] = np.append(train_Y_iter[0], to_add_y)

 Deleting the point from the pool.

pool_X = np.delete(pool_X, in_var, axis=0)
pool_Y = np.delete(pool_Y, in_var)

### Training again and visualising the results (Iteration 1)

This time, we will pass previously learnt prior to the next iteration.

S0 = np.eye(N_features)
models[1] = BLR(S0, models[0].MN)

models[1].fit(train_X_iter[1], train_Y_iter[1])

estimations[1], _ = models[1].predict(test_X)
test_mae_error[1] = np.mean(np.abs(test_Y - estimations[1]))

MAE on the test set is

test_mae_error[1]

Visualizing the results.

fig, ax = plt.subplots()
plot(ax, models[1])

Before & after adding most uncertain point

fig, ax = plt.subplots(1,2, figsize=(13.5,4.5), sharey=True)
plt.subplots_adjust(wspace=0.03)
plot(ax[0], models[0],'Before')
plot(ax[1], models[1],'After')

We can see that including most uncertain point into the train set has produced a better fit and MAE for test set has been reduced. Also, uncertainty has reduced at the left part of the data but it has increased a bit on the right part of the data.

Now let's do this for few more iterations in a loop and visualise the results.

### Active learning procedure

num_iterations = 20
points_added_x= np.zeros((num_iterations+1, N_features))

points_added_y=[]

print("Iteration, Cost\n")
print("-"*40)

for iteration in range(2, num_iterations+1):
    # Making predictions on the pool set based on model learnt in the respective train set 
    estimations_pool, var = models[iteration-1].predict(pool_X)
    
    # Finding the point from the pool with highest uncertainty
    in_var = var.diagonal().argmax()
    to_add_x = pool_X[in_var,:]
    to_add_y = pool_Y[in_var]
    points_added_x[iteration-1,:] = to_add_x
    points_added_y.append(to_add_y)
    
    # Adding the point to the train set from the pool
    train_X_iter[iteration] = np.vstack([train_X_iter[iteration-1], to_add_x])
    train_Y_iter[iteration] = np.append(train_Y_iter[iteration-1], to_add_y)
    
    # Deleting the point from the pool
    pool_X = np.delete(pool_X, in_var, axis=0)
    pool_Y = np.delete(pool_Y, in_var)
    
    # Training on the new set
    models[iteration] = BLR(S0, models[iteration-1].MN)
    models[iteration].fit(train_X_iter[iteration], train_Y_iter[iteration])
    
    estimations[iteration], _ = models[iteration].predict(test_X)
    test_mae_error[iteration]= pd.Series(estimations[iteration] - test_Y.squeeze()).abs().mean()
    print(iteration, (test_mae_error[iteration]))

pd.Series(test_mae_error).plot(style='ko-')
plt.xlim((-0.5, num_iterations+0.5))
plt.ylabel("MAE on test set")
plt.xlabel("\# Points Queried")
plt.show()

The plot above shows that MAE on the test set fluctuates a bit initially then reduces gradually as we keep including more points from the pool to the train set. Let's visualise fits for all the iterations. We'll discuss this behaviour after that.

### Visualizing active learning procedure

print('Initial model')
print('Y = {0:0.2f} X^5 + {1:0.2f} X^4 + {2:0.2f} X^3 + {3:0.2f} X^2 + {4:0.2f} X + {5:0.2f}'.format(*models[0].MN[::-1]))
print('\nFinal model')
print('Y = {0:0.2f} X^5 + {1:0.2f} X^4 + {2:0.2f} X^3 + {3:0.2f} X^2 + {4:0.2f} X + {5:0.2f}'.format(*models[num_iterations].MN[::-1]))

def update(iteration):
    ax.cla()
    plot(ax, models[iteration])
    fig.tight_layout()

fig, ax = plt.subplots(figsize=(10,5))
anim = FuncAnimation(fig, update, frames=np.arange(0, num_iterations+1, 1), interval=250)
plt.close()
rc('animation', html='jshtml')

anim

We can see that the point having highest uncertainty was chosen in first iteration and it produced the near optimal fit. After that, error reduced gradually. 

Now, let's put everything together and create a class for active learning procedure

### Creating a class for active learning procedure

class ActiveL():
  def __init__(self, X, y, S0=None, M0=None, test_size=0.5, degree = 5, iterations = 20, seed=1):
    self.X_init = X
    self.y = y
    self.S0 = S0
    self.M0 = M0
    self.train_X_iter = {} # to store train points at each iteration
    self.train_Y_iter = {} # to store corresponding labels to the train set at each iteration
    self.models = {} # to store the models at each iteration
    self.estimations = {} # to store the estimations on the test set at each iteration
    self.test_mae_error = {} # to store MAE(Mean Absolute Error) at each iteration
    self.test_size = test_size
    self.degree = degree
    self.iterations = iterations
    self.seed = seed
    self.train_size = degree + 2

  def data_preperation(self):
    # Adding polynomial features
    self.X = PolynomialFeatures(degree=self.degree).fit_transform(self.X_init)
    N_features = self.X.shape[1]
    
    # Splitting into train, test and pool
    train_pool_X, self.test_X, train_pool_Y, self.test_Y = train_test_split(self.X, self.y, 
                                                                            test_size=self.test_size,
                                                                            random_state=self.seed)
    self.train_X, self.pool_X, self.train_Y, self.pool_Y = train_test_split(train_pool_X, train_pool_Y, 
                                                                            train_size=self.train_size, 
                                                                            random_state=self.seed)
    
    # Setting BLR prior incase of not given
    if self.M0 == None:
      self.M0 = np.zeros((N_features, ))
    if self.S0 == None:
      self.S0 = np.eye(N_features)
    
  def main(self):
    # Training for iteration 0
    self.train_X_iter[0] = self.train_X
    self.train_Y_iter[0] = self.train_Y
    self.models[0] = BLR(self.S0, self.M0)
    self.models[0].fit(self.train_X, self.train_Y)

    # Running loop for all iterations
    for iteration in range(1, self.iterations+1):
      # Making predictions on the pool set based on model learnt in the respective train set 
      estimations_pool, var = self.models[iteration-1].predict(self.pool_X)
      
      # Finding the point from the pool with highest uncertainty
      in_var = var.diagonal().argmax()
      to_add_x = self.pool_X[in_var,:]
      to_add_y = self.pool_Y[in_var]
      
      # Adding the point to the train set from the pool
      self.train_X_iter[iteration] = np.vstack([self.train_X_iter[iteration-1], to_add_x])
      self.train_Y_iter[iteration] = np.append(self.train_Y_iter[iteration-1], to_add_y)
      
      # Deleting the point from the pool
      self.pool_X = np.delete(self.pool_X, in_var, axis=0)
      self.pool_Y = np.delete(self.pool_Y, in_var)
      
      # Training on the new set
      self.models[iteration] = BLR(self.S0, self.models[iteration-1].MN)
      self.models[iteration].fit(self.train_X_iter[iteration], self.train_Y_iter[iteration])
      
      self.estimations[iteration], _ = self.models[iteration].predict(self.test_X)
      self.test_mae_error[iteration]= pd.Series(self.estimations[iteration] - self.test_Y.squeeze()).abs().mean()

  def _plot_iter_MAE(self, ax, iteration):
    ax.plot(list(self.test_mae_error.values())[:iteration+1], 'ko-')
    ax.set_title('MAE on test set over iterations')
    ax.set_xlim((-0.5, self.iterations+0.5))
    ax.set_ylabel("MAE on test set")
    ax.set_xlabel("\# Points Queried")
  
  def _plot(self, ax, model):
    # Plotting the pool
    ax.scatter(self.pool_X[:,1], self.pool_Y, label='pool',s=1,color='r',alpha=0.4)
    
    # Plotting the test data
    ax.scatter(self.test_X[:,1], self.test_Y, label='test data',s=1, color='b', alpha=0.4)
    
    # Combining test_pool
    test_pool_X, test_pool_Y = np.append(self.test_X, self.pool_X, axis=0), np.append(self.test_Y, self.pool_Y)
    
    # Sorting test_pool
    sorted_inds = np.argsort(test_pool_X[:,1])
    test_pool_X, test_pool_Y = test_pool_X[sorted_inds], test_pool_Y[sorted_inds]
    
    # Plotting test_pool with uncertainty
    preds, var = model.predict(test_pool_X)
    individual_var = var.diagonal()
    ax.plot(test_pool_X[:,1], model.y_hat_map, color='black', label='model')
    ax.fill_between(test_pool_X[:,1], model.y_hat_map-individual_var, model.y_hat_map+individual_var
                    , alpha=0.2, color='black', label='uncertainty')
    
    # plotting the train data
    ax.scatter(model.x[:,1], model.y,s=10, color='k', marker='s', label='train data')
    ax.scatter(model.x[-1,1], model.y[-1],s=80, color='r', marker='o', label='last added point')
    
    # plotting MAE
    preds, var = model.predict(self.test_X)
    ax.set_title('MAE is '+str(np.mean(np.abs(self.test_Y - preds))))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
  def visualise_AL(self):
    fig, ax = plt.subplots(1,2,figsize=(13,5))
    def update(iteration):
      ax[0].cla()
      ax[1].cla()
      self._plot(ax[0], self.models[iteration])
      self._plot_iter_MAE(ax[1], iteration)
      fig.tight_layout()

    print('Initial model')
    print('Y = '+' + '.join(['{0:0.2f}'.format(self.models[0].MN[i])+' X^'*min(i,1)+str(i)*min(i,1) for i in range(self.degree+1)]))
    print('\nFinal model')
    print('Y = '+' + '.join(['{0:0.2f}'.format(self.models[self.iterations].MN[i])+' X^'*min(i,1)+str(i)*min(i,1) for i in range(self.degree+1)]))

    anim = FuncAnimation(fig, update, frames=np.arange(0, self.iterations+1, 1), interval=250)
    plt.close()

    rc('animation', html='jshtml')
    return anim

### Visualizing a different polynomial fit on the same dataset

Let's try to fit a degree 7 polynomial to the same data now.

np.random.seed(seed)
X_init = np.linspace(-1, 1, 1000)
noise = np.random.randn(1000, ) * 0.5
Y = (5 * X_init**3 - 4 * X_init**2 + 3 * X_init - 2) + noise

model = ActiveL(X_init.reshape(-1,1), Y, degree=7, iterations=20, seed=seed)

model.data_preperation()
model.main()
model.visualise_AL()

We can clearly see that model was fitting the train points well and uncertainty was high at the left-most position. After first iteration, the left-most point was added to the train set and MAE reduced significantly. Similar phenomeneon happened at iteration 2 with the right-most point. After that error kept reducing at slower rate gradually because fit was near optimal after just 2 iterations.

### Active learning for diabetes dataset from the Scikit-learn module

Let's run our model for diabetes data from sklearn module. The data have various features like age, sex, weight etc. of diabetic people and target is increment in disease after one year. We'll choose only 'weight' feature, which seems to have more correlation with the target.

We'll try to fit degree 1 polynomial to this data, as our data seems to have a linear fit. First, let's check the performance of Scikit-learn linear regression model.

X, Y = datasets.load_diabetes(return_X_y=True)
X = X[:, 2].reshape(-1,1) # Choosing only feature 2 which seems more relevent to linear regression

# Normalizing
X = (X - X.min())/(X.max() - X.min())
Y = (Y - Y.min())/(Y.max() - Y.min())

Visualizing the dataset.

plt.scatter(X, Y)
plt.xlabel('Weight of the patients')
plt.ylabel('Increase in the disease after a year')
plt.show()

Let's fit the Scikit-learn linear regression model with 50% train-test split.

from sklearn.linear_model import LinearRegression
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state = seed)

clf = LinearRegression()

clf.fit(train_X, train_Y)
pred_Y = clf.predict(test_X)

Visualizing the fit & MAE.

plt.scatter(X, Y, label='data', s=5)
plt.plot(test_X, pred_Y, label='model', color='r')
plt.xlabel('Weight of the patients')
plt.ylabel('Increase in the disease after a year')
plt.title('MAE is '+str(np.mean(np.abs(pred_Y - test_Y))))
plt.legend()
plt.show()

Now we'll fit the same data with our BLR model 

model = ActiveL(X.reshape(-1,1), Y, degree=1, iterations=20, seed=seed)

model.data_preperation()
model.main()
model.visualise_AL()

Initially, the fit is leaning towards zero slope, which is the influence of bias due to a low number of training points. It's interesting to see that our initial train points tend to make a vertical fit, but the model doesn't get carried away by that and stabilizes the self with prior.

print('MAE for Scikit-learn Linear Regression is',np.mean(np.abs(pred_Y - test_Y)))
print('MAE for Bayesian Linear Regression is', model.test_mae_error[20])

At the end, results of sklearn linear regression and our active learning based BLR model are comparable even though we've used only 20 points to train our model over 221 points used by sklearn. This is because active learning enables us to choose those datapoints for training, which are going to contribute the most towards a precise fit.