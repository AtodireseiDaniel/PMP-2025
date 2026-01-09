import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)
x_raw = np.linspace(-2, 2, 500)
y_raw = 0.5 * x_raw**2 - x_raw + 0.2 + np.random.normal(0, 0.4, 500)

order = 5
x_p = np.vstack([x_raw**i for i in range(1, order+1)])
x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
y_s = (y_raw - y_raw.mean()) / y_raw.std()

sd_configs = [10, 100, np.array([10, 0.1, 0.1, 0.1, 0.1])]
labels = ['sd=10', 'sd=100', 'sd=vector']
idatas_500 = []

for sd_val in sd_configs:
    with pm.Model() as model_500:
        alpha = pm.Normal('a', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=sd_val, shape=order)
        epsilon = pm.HalfNormal('eps', 5)
        mu = alpha + pm.math.dot(beta, x_s)
        pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_s)
        idata = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.9)
        idatas_500.append(idata)

plt.figure(figsize=(10, 6))
plt.scatter(x_s[0], y_s, c='gray', alpha=0.2, label='500 Date')

for i, idata in enumerate(idatas_500):
    a_post = idata.posterior['a'].mean(("chain", "draw")).values
    b_post = idata.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_s[0])
    y_post = a_post + np.dot(b_post, x_s)
    plt.plot(x_s[0][idx], y_post[idx], label=labels[i], linewidth=2)

plt.legend()
plt.show()