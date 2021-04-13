# Eigen values

In this notebook, we will look at eigen values

https://www.youtube.com/watch?v=PFDu9oVAE-g    
    

import numpy as np
import matplotlib.pyplot as plt

plt.arrow(0, 0, 1, 1,shape='full', head_width=0.05, head_length=0.1)
plt.scatter(0, 0)
plt.grid()
plt.ylim((-1, 3))
plt.xlim(-1, 3)

def m_lam(M, lam):
    M[0, 0] = M[0, 0] - lam
    M[1, 1] = M[1, 1] - lam
    return M
    

def plot_transformation(x, M):
    arrow_in = plt.arrow(0, 0, x[0], x[1],shape='full', head_width=0.05, head_length=0.1,color='green',lw=8, alpha=0.6)
    plt.scatter(0, 0)
    plt.grid()
    Mx = M@x


    plt.ylim((min(Mx[1], x[1], 0)-1, max(Mx[1], x[1], 0)+1))
    plt.xlim((min(Mx[0], x[0], 0)-1, max(Mx[0], x[0], 0)+1))


    arrow_out = plt.arrow(0, 0, Mx[0], Mx[1],shape='full', head_width=0.05, head_length=0.1, color='red')
    plt.legend([arrow_in, arrow_out, ], ['Input','Transformed',], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal')
    return Mx

plot_transformation([2, 1], np.array([[3, 1], [0, 2]]))


plot_transformation([-2, 1], m_lam(np.array([[3, 1], [0, 2]]), 2))


for i in range(-5, 5, 1):
    for j in range(-5, 5, 1):
        plot_transformation([-i, j], m_lam(np.array([[3, 1], [0, 2]]), 2))

for i in range(-5, 5, 1):
    for j in range(-5, 5, 1):
        plot_transformation([-i, j], m_lam(np.array([[3, 1], [0, 2]]), 3))

For these two values of lambda (eigen values), any input vector will be squished to the two red lines shown (determinant is zero, thus ..). 

For lambda = 3, we have

[[3-3, 1], [0, -1]](x, y) = 0

i.e. y = 0

For lambda =2, we have

[[3-2, 1], [0, 2-2]](x, y) = 0

i.e. y = -x



Let us take an input vector [1, -1] and see it under M



plot_transformation([1, -1], np.array([[3, 1], [0, 2]]))

plot_transformation([-1, 1], np.array([[3, 1], [0, 2]]))

In above two, we can see that the input vectors become two times (=eigen value)