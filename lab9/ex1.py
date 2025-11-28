import os

os.environ["PYTENSOR_FLAGS"] = "cxx="
# am avut problema cu MIN_GW si de asta am astea
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt



y_vals = [0, 5, 10]
thetas = [0.2, 0.5]

fig1, ax1 = plt.subplots(3, 2, figsize=(12, 12))
fig1.suptitle("A")

fig2, ax2 = plt.subplots(3, 2, figsize=(12, 12))
fig2.suptitle("C ")

print("Y_obs   Theta   Media n   Interval 94%")
print("----------------------------------------")

for i, y in enumerate(y_vals):
    for j, p in enumerate(thetas):
        with pm.Model() as model:
            n = pm.Poisson("n", mu=10, initval=np.max([10, y + 5]))
            obs = pm.Binomial("obs", n=n, p=p, observed=y)

            idata = pm.sample(2000, tune=1000, chains=2, cores=1, return_inferencedata=True, progressbar=False)

            pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True, progressbar=False)

        stats = az.summary(idata, hdi_prob=0.94)
        medie = stats.loc['n', 'mean']
        lim_inf = stats.loc['n', 'hdi_3%']
        lim_sup = stats.loc['n', 'hdi_97%']

        print(f"{y}        {p}        {medie:.2f}       [{lim_inf:.0f}, {lim_sup:.0f}]")

        az.plot_posterior(idata, var_names=["n"], ax=ax1[i, j], hdi_prob=0.94)
        ax1[i, j].set_title(f" Y={y}, theta={p}")

        post_pred = idata.posterior_predictive["obs"]
        az.plot_dist(post_pred, ax=ax2[i, j], color="orange", label="Predictie")
        ax2[i, j].axvline(y, color="k", linestyle="--", label="Observat")
        ax2[i, j].set_title(f"Predictie Y* (Y={y}, t={p})")
        ax2[i, j].legend()

plt.tight_layout()
plt.show()