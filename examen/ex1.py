import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv('bike_daily.csv')
sns.pairplot(df[['temp_c', 'humidity', 'wind_kph', 'is_holiday', 'season', 'rentals']])

scaler = StandardScaler()
continuous_predictors = ['temp_c', 'humidity', 'wind_kph']
X = df[continuous_predictors].values
y = df['rentals'].values

X_scaled = scaler.fit_transform(X)
y_scaled = (y - np.mean(y)) / np.std(y)

with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_temp = pm.Normal('beta_temp', mu=0, sigma=10)
    beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=10)
    beta_wind = pm.Normal('beta_wind', mu=0, sigma=10)
    beta_is_holiday = pm.Normal('beta_is_holiday', mu=0, sigma=10)
    beta_season = pm.Normal('beta_season', mu=0, sigma=10)

    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta_temp * X_scaled[:, 0] + beta_humidity * X_scaled[:, 1] + \
         beta_wind * X_scaled[:, 2] + beta_is_holiday * df['is_holiday'] + \
         beta_season * df['season']

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)

    trace_linear = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True)

pm.summary(trace_linear)

with pm.Model() as model_polynomial:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_temp = pm.Normal('beta_temp', mu=0, sigma=10)
    beta_temp2 = pm.Normal('beta_temp2', mu=0, sigma=10)
    beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=10)
    beta_wind = pm.Normal('beta_wind', mu=0, sigma=10)
    beta_is_holiday = pm.Normal('beta_is_holiday', mu=0, sigma=10)
    beta_season = pm.Normal('beta_season', mu=0, sigma=10)

    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta_temp * X_scaled[:, 0] + beta_temp2 * X_scaled[:, 0] ** 2 + \
         beta_humidity * X_scaled[:, 1] + beta_wind * X_scaled[:, 2] + \
         beta_is_holiday * df['is_holiday'] + beta_season * df['season']

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)

    trace_polynomial = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True)

pm.summary(trace_polynomial)

y_pred_linear = trace_linear['alpha'].mean() + trace_linear['beta_temp'].mean() * X_scaled[:, 0] + \
                trace_linear['beta_humidity'].mean() * X_scaled[:, 1] + trace_linear['beta_wind'].mean() * X_scaled[
                    :, 2] + \
                trace_linear['beta_is_holiday'].mean() * df['is_holiday'] + trace_linear['beta_season'].mean() * df[
                    'season']

y_pred_polynomial = trace_polynomial['alpha'].mean() + trace_polynomial['beta_temp'].mean() * X_scaled[:, 0] + \
                    trace_polynomial['beta_temp2'].mean() * X_scaled[:, 0] ** 2 + trace_polynomial[
                        'beta_humidity'].mean() * X_scaled[:, 1] + \
                    trace_polynomial['beta_wind'].mean() * X_scaled[:, 2] + trace_polynomial['beta_is_holiday'].mean() * \
                    df['is_holiday'] + \
                    trace_polynomial['beta_season'].mean() * df['season']

print(classification_report(y_scaled, y_pred_linear > 0.5))
print(classification_report(y_scaled, y_pred_polynomial > 0.5))

waic_linear = az.waic(trace_linear, model=model_linear)
waic_polynomial = az.waic(trace_polynomial, model=model_polynomial)

loo_linear = az.loo(trace_linear, model=model_linear)
loo_polynomial = az.loo(trace_polynomial, model=model_polynomial)

print(waic_linear)
print(waic_polynomial)

print(loo_linear)
print(loo_polynomial)

with model_linear:
    posterior_predictive_linear = pm.sample_posterior_predictive(trace_linear, var_names=['y_obs'], samples=2000)

with model_polynomial:
    posterior_predictive_polynomial = pm.sample_posterior_predictive(trace_polynomial, var_names=['y_obs'], samples=2000)