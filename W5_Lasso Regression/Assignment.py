import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)


sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms'] * sales['bedrooms']
sales['floors_square'] = sales['floors'] * sales['floors']

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms'] * testing['bedrooms']
testing['floors_square'] = testing['floors'] * testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms'] * training['bedrooms']
training['floors_square'] = training['floors'] * training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms'] * validation['bedrooms']
validation['floors_square'] = validation['floors'] * validation['floors']

all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']


#########################################################################################################

model_all = Lasso(alpha=5e2, normalize=True)
model_all.fit(sales[all_features], sales['price'])

alphas = np.logspace(1, 7, num=13)
rss_vals = []

for alpha in alphas:
  model = Lasso(alpha=alpha, normalize=True)
  model.fit(training[all_features], training['price'])
  predicted = model.predict(validation[all_features])
  error = predicted - validation['price']
  rss = (error**2).sum()
  print(alpha, rss)
  rss_vals.append((rss, alpha))

best_alpha = min(rss_vals)[1]
print(best_alpha)
best_model = Lasso(alpha=best_alpha, normalize=True)
best_model.fit(training[all_features], training['price'])
predicted = best_model.predict(testing[all_features])
error = predicted - testing['price']
rss_test = (error**2).sum()
print(rss_test)
print(best_model.intercept_, best_model.coef_)
not_zero = np.count_nonzero(best_model.coef_) + np.count_nonzero(best_model.intercept_)
print(not_zero)

#########################################################################################################

max_nonzeros = 7

alphas = np.logspace(1, 4, num=20)
non_zeros = []
for alpha in alphas:
  model = Lasso(alpha=alpha, normalize=True)
  model.fit(training[all_features], training['price'])
  nz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
  non_zeros.append((alpha, nz))
print(non_zeros)


#########################################################################################################

l1_penalty_min = 127.42749857031335
l1_penalty_max = 263.66508987303581
alphas = np.linspace(l1_penalty_min, l1_penalty_max, 20)
rss_vals = []
for alpha in alphas:
  model = Lasso(alpha=alpha, normalize=True)
  model.fit(training[all_features], training['price'])
  nz = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
  if nz == max_nonzeros:
    predicted = model.predict(validation[all_features])
    error = predicted - validation['price']
    rss = (error**2).sum()
    rss_vals.append((rss, alpha))
print(rss_vals)
