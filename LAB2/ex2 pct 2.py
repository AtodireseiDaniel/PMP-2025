import numpy as np
import matplotlib.pyplot as plt
lambdas=[1,2,5,10]
n= 1000
random_lambda=np.random.choice(lambdas,n,replace=True)
poisson_random=np.random.poisson(random_lambda)
plt.hist(poisson_random, bins=range(0,max(poisson_random)+1), color='orange', edgecolor='black', alpha=0.7)
plt.title(f"distributia poisson {lambdas}")
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

plt.hist(random_lambda, bins=[0.5,1.5,2.5,5.5,10.5], rwidth=0.8, edgecolor='black')
plt.title("Distribuția λ alese aleator")
plt.xlabel("λ ales")
plt.ylabel("Frecvență")
plt.show()