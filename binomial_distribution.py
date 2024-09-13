import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

def fac(n):
    value = 1
    for i in range(2, n+1):
        value = value * i

    return value

def C(n, k, p):
    return fac(n) / (fac(k) * (fac(n-k))) * pow(p, k) * pow(1-p, n-k)

'''
n = 10
p = 0.5

x = random.binomial(n=n, p=p, size=1000)

sns.distplot(x, hist=True, kde=False)
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.show()
'''

for N in [10, 20, 30]:
    X = []
    Y = []

    for K in range(1, 100):
        X.append(K)
        Y.append( C(N, K, 0.5) )

    plt.plot(X, Y)

plt.legend( ('N = 10', 'N = 20', 'N = 30') )
plt.show()

