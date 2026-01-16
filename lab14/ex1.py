import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pytensor.tensor as pt

df = pd.read_csv('date_colesterol.csv')
t = df['Ore_Exercitii'].values
y_obs = df['Colesterol'].values

clusters = [3, 4, 5]
idatas = {}
models = {}

for K in clusters:
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(K))

        init_alpha = np.sort(np.linspace(y_obs.min(), y_obs.max(), K))

        alpha = pm.Normal('alpha', mu=y_obs.mean(), sigma=20, shape=K, initval=init_alpha)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=K)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=K)

        sigma = pm.HalfNormal('sigma', sigma=10, shape=K)

        pm.Potential('order_alpha', pt.switch(pt.any(pt.diff(alpha) <= 0), -np.inf, 0))

        mu = alpha + beta * t[:, None] + gamma * (t[:, None] ** 2)

        y = pm.NormalMixture('y', w=w, mu=mu, sigma=sigma, observed=y_obs)

        idata = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=123, return_inferencedata=True)
        idatas[K] = idata
        models[K] = model

for K in clusters:
    print(f"\n--- Estimari Parametri K={K} ---")
    print(az.summary(idatas[K], var_names=['w', 'alpha', 'beta', 'gamma']))

comp = az.compare({str(K): idatas[K] for K in clusters}, ic="waic", scale="deviance")
print("\n--- Comparare Modele (WAIC) ---")
print(comp)
az.plot_compare(comp)