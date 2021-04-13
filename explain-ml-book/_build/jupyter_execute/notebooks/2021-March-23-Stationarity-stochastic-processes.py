import numpy as np
import matplotlib.pyplot as plt

def AR(phi, n):
    order = len(phi)
    x = [0 for _ in range(order)]
    for i in range(n-order):
        tmp = np.sum(np.array(x[-order:])*phi+np.random.normal(0,1))
        x.append(tmp)
    return x

for i in range(1000):
    X = AR([0.8], 100)
    plt.plot(X)

for i in range(1000):
    X = AR([0.99], 100)
    plt.plot(X)

for i in range(1000):
    X = AR([1], 100)
    plt.plot(X)

for i in range(1000):
    X = AR([1.01], 100)
    plt.plot(X)

