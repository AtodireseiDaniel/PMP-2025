import numpy as np
import matplotlib.pyplot as plt
lambdas = [1,2,5,10]

n = 1000

for lam in lambdas:
    poisson_random = np.random.poisson(lam,n)
    plt.hist(poisson_random, bins=15, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f"distributia poisson {lam}")
    plt.xlabel('Valori')
    plt.ylabel('Frecvență')
    plt.show()