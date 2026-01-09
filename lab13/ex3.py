import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

dummy_data = np.loadtxt('dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
y_1s = (y_1 - y_1.mean()) / y_1.std()


def run_model(order):
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

    with pm.Model() as model:
        alpha = pm.Normal('a', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('eps', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.9)
        pm.compute_log_likelihood(idata)
    return idata


idata_l = run_model(1)
idata_p2 = run_model(2)
idata_p3 = run_model(3)

compare_dict = {'linear': idata_l, 'quadratic': idata_p2, 'cubic': idata_p3}

comp_waic = az.compare(compare_dict, ic="waic", scale="deviance")
print(comp_waic)

comp_loo = az.compare(compare_dict, ic="loo", scale="deviance")
print(comp_loo)

az.plot_compare(comp_waic)
plt.show()