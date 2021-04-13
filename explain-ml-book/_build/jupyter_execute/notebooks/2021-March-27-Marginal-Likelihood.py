import numpy as np
import matplotlib.pyplot as plt

Assume we have a discrete random variable X describing the sum of the numbers on the top of two die

We have the following distribution

\begin{array}{|l|l|}
\hline x & \mathrm{P}(\mathrm{X}=x) \\
\hline 2 & 1 / 36 \\
\hline 3 & 2 / 36 \\
\hline 4 & 3 / 36 \\
\hline 5 & 4 / 36 \\
\hline 6 & 5 / 36 \\
\hline 7 & 6 / 36 \\
\hline 8 & 5 / 36 \\
\hline 9 & 4 / 36 \\
\hline 10 & 3 / 36 \\
\hline 11 & 2 / 36 \\
\hline 12 & 1 / 36 \\
\hline
\end{array}



import pandas as pd

df = pd.DataFrame({"p(X=x)":[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], "x":range(2, 13)})
df["p(X=x)"] = df["p(X=x)"]/36
df.index = df['x']
df

df.plot(kind='bar', subplots=True)

We can calculate the $\mathbb{E}(X) = \sum x p(X=x)$

$\mathbb{E}(p(D|\theta)) = \int p(D|theta) p(\theta)d\theta$

$\mathbb{E}(p(D|\theta)) = \sum_{i=1}^N p(D|\theta_i) p(\theta=\theta_i)$ where N is a large number



(df["p(X=x)"]*df["x"]).sum()

from scipy.stats import multivariate_normal, norm

n = norm(loc=0, scale=1)
M = multivariate_normal()

n.pdf([0, 1])

