import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split

import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


def polynomial_dataframe(feature, degree):
  df = pd.DataFrame()
  df['power_1'] = feature
  if degree > 1:
    for power in range(2, degree + 1):
      name = 'power_' + str(power)
      df[name] = feature.apply(lambda x: x**power)
  return df


def get_features(d):
  features = []
  for i in range(1, d + 1):
    features.append('power_' + str(i))
  return features


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


# load and sort data
train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
cv_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)

train_data = train_data.sort_values(['sqft_living', 'price'])
cv_data = cv_data.sort_values(['sqft_living', 'price'])
test_data = test_data.sort_values(['sqft_living', 'price'])
train_valid_shuffled = train_valid_shuffled.sort_values(['sqft_living', 'price'])

all_data = pd.concat([train_data, cv_data, test_data])
all_data = all_data.sort_values(['sqft_living', 'price'])

######################################################################################################################


l2_small_penalty = 0.01
features = ['power_1', 'power_2']
poly_data_2 = polynomial_dataframe(all_data['sqft_living'], 2)
ml_ridge = Ridge(alpha=l2_small_penalty, normalize=True)
ml_ridge.fit(poly_data_2[features], all_data['price'])
plt.plot(all_data['sqft_living'], all_data['price'], '.', all_data['sqft_living'], ml_ridge.predict(poly_data_2[features]), '-')
plt.show()

X = poly_data_2[features]
ml = LinearRegression()
ml.fit(X, all_data['price'])
plt.plot(all_data['sqft_living'], all_data['price'], '.', all_data['sqft_living'], ml.predict(X), '-')
plt.show()


l2_small_penalty = 1.5e-5
features = get_features(15)
poly15_data = polynomial_dataframe(train_data['sqft_living'], 15)
X_train = poly15_data[features]
y_train = train_data['price']
model = Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(X_train, y_train)
predicted = model.predict(X_train)
print(model.intercept_, model.coef_)
print(model.score(X_train, y_train))
plt.plot(train_data['sqft_living'], y_train, '.', train_data['sqft_living'], predicted, '-')
plt.show()

######################################################################################################################


def plot_model(data, alpha):
  data = data.sort_values(['sqft_living', 'price'])
  df = polynomial_dataframe(data['sqft_living'], 15)

  ml1 = LinearRegression()
  ml1.fit(df, data['price'])
  predicted = ml1.predict(df)
  print(ml1.intercept_, ml1.coef_)
  print(ml1.score(df, data['price']))
  plt.plot(data['sqft_living'], data['price'], '.', data['sqft_living'], predicted, '-')
  plt.show()

  ml2 = Ridge(alpha=alpha, normalize=True)
  ml2.fit(df, data['price'])
  predicted = ml2.predict(df)
  print(ml2.intercept_, ml2.coef_)
  print(ml2.score(df, data['price']))
  plt.plot(data['sqft_living'], data['price'], '.', data['sqft_living'], predicted, '-')
  plt.show()


def try_sets(alpha):
  set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
  set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
  set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
  set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
  plot_model(set_1, alpha)
  plot_model(set_2, alpha)
  plot_model(set_3, alpha)
  plot_model(set_4, alpha)


l2_penalty = 1e-9
try_sets(l2_penalty)
l2_penalty = 1.23e2
try_sets(l2_penalty)

######################################################################################################################


def k_fold_cross_validation(k, alpha, X, y):
  rss_avg = 0
  n = X.shape[0]
  for i in xrange(k):
    start = (n * i) / k
    end = (n * (i + 1)) / k - 1
    cv_X = X[start:end + 1]
    cv_y = y[start:end + 1]
    train_X = X[0:start].append(X[end + 1:n])
    train_y = y[0:start].append(y[end + 1:n])

    model = Ridge(alpha=alpha, normalize=True)
    model.fit(train_X, train_y)
    predicted = model.predict(cv_X)
    error = predicted - cv_y
    rss = (error**2).sum()
    rss_avg += rss
  return rss_avg / k


df_k_fold = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
df_test = polynomial_dataframe(test_data['sqft_living'], 15)

alphas = np.logspace(3, 9, num=13)
features = get_features(15)

X_k_fold = df_k_fold[features]
y_k_fold = train_valid_shuffled['price']
X_test = df_test[features]
y_test = test_data['price']

alpha_rss = []
for alpha in alphas:
  rss_avg = k_fold_cross_validation(10, alpha, X_k_fold, y_k_fold)
  alpha_rss.append((rss_avg, alpha))
print(min(alpha_rss))

best_alpha = min(alpha_rss)[1]
final_model = Ridge(alpha=best_alpha, normalize=True)
final_model.fit(X_k_fold, y_k_fold)
predicted = final_model.predict(X_test)
error = predicted - y_test
rss = (error**2).sum()
print(rss)
