import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfCauchy('sigma', beta=5)

    mu = pm.Deterministic('mu', alpha + beta * publicity)

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=sales)

    idata = pm.sample(2000, tune=1000, return_inferencedata=True, progressbar=False)

print(az.summary(idata, var_names=['alpha', 'beta'], hdi_prob=0.95)[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']])

plt.figure(figsize=(10, 6))
plt.scatter(publicity, sales, color='blue')

post = idata.posterior
alpha_mean = post['alpha'].mean().item()
beta_mean = post['beta'].mean().item()

plt.plot(publicity, alpha_mean + beta_mean * publicity, color='red')
az.plot_hdi(publicity, post['mu'], hdi_prob=0.95, color='gray')

plt.xlabel('Publicity')
plt.ylabel('Sales')
plt.show()