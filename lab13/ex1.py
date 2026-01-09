import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def main():
    # 1. Pregatire date (pag. 5)
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]


    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    # 2. Configuratii SD pentru Ex 1a si 1b
    sd_configs = [10, 100, np.array([10, 0.1, 0.1, 0.1, 0.1])]
    labels = ['Ex 1a: sd=10', 'Ex 1b: sd=100', 'Ex 1b: sd=vector']
    idatas = []

    # 3. Bucla de inferenta (pag. 6)
    for sd_val in sd_configs:
        with pm.Model() as model_p:
            alpha = pm.Normal('a', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=sd_val, shape=order)
            epsilon = pm.HalfNormal('eps', 5)
            mu = alpha + pm.math.dot(beta, x_1s)
            y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)

            # Eșantionare - cores=1 poate ajuta daca eroarea persista
            idata = pm.sample(2000, return_inferencedata=True, progressbar=True)
            idatas.append(idata)

    # 4. Plotare (pag. 7)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_1s[0], y_1s, c='black', marker='.', label='Date reale')

    for i, idata in enumerate(idatas):
        a_post = idata.posterior['a'].mean(("chain", "draw")).values
        b_post = idata.posterior['beta'].mean(("chain", "draw")).values
        idx = np.argsort(x_1s[0])
        y_post = a_post + np.dot(b_post, x_1s)
        plt.plot(x_1s[0][idx], y_post[idx], label=labels[i])

    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()