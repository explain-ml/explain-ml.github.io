In this notebook, we will look at determinants

https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=6
    
    

import numpy as np
import matplotlib.pyplot as plt

plt.arrow(0, 0, 1, 1,shape='full', head_width=0.05, head_length=0.1)
plt.scatter(0, 0)
plt.grid()
plt.ylim((-1, 3))
plt.xlim(-1, 3)

def plot_transformation(x, M):
    arrow_in = plt.arrow(0, 0, x[0], x[1],shape='full', head_width=0.05, head_length=0.1,color='green')
    plt.scatter(0, 0)
    plt.grid()
    Mx = M@x


    plt.ylim((min(Mx[1], x[1], 0)-1, max(Mx[1], x[1], 0)+1))
    plt.xlim((min(Mx[0], x[0], 0)-1, max(Mx[0], x[0], 0)+1))


    arrow_out = plt.arrow(0, 0, Mx[0], Mx[1],shape='full', head_width=0.05, head_length=0.1, color='red')
    plt.legend([arrow_in, arrow_out, ], ['Input','Transformed',], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal')
    return Mx

plot_transformation([1, 1], np.array([[1, 0], [0, 2]]))

plot_transformation([1, 1], np.array([[2, 3], [4, 6]]))
plot_transformation([2, 1], np.array([[2, 3], [4, 6]]))

for i in range(-5, 5, 1):
    for j in range(-5, 5, 1):
        plot_transformation([i, j], np.array([[-2, -3], [4, 6]]))

Because the matrix is not full rank, x(-2, 3) + y(4, 6) for any x and y will be along y=1.5x line

