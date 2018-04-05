import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
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


######################################################################################################################


# Features type
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

# load datasets
train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
cv_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

# sort data
train_data = train_data.sort_values(['sqft_living', 'price'])
cv_data = cv_data.sort_values(['sqft_living', 'price'])
test_data = test_data.sort_values(['sqft_living', 'price'])

# label column
y_train = train_data['price']
y_cv = cv_data['price']
y_test = test_data['price']


######################################################################################################################


features = ['power_1']
polydata1 = polynomial_dataframe(train_data['sqft_living'], 1)
polydata1['price'] = y_train
X_train = polydata1[features]
ml1 = LinearRegression()
ml1.fit(X_train, y_train)
predict_train = ml1.predict(X_train)
print(ml1.intercept_, ml1.coef_)
plt.plot(X_train, y_train, '.', X_train, predict_train, '-')
plt.show()


features.append('power_2')
polydata2 = polynomial_dataframe(train_data['sqft_living'], 2)
polydata2['price'] = y_train
X_train_2 = polydata2[features]
ml2 = LinearRegression()
ml2.fit(X_train_2, y_train)
predict_train = ml2.predict(X_train_2)
print(ml2.intercept_, ml2.coef_)
plt.plot(X_train, y_train, '.', X_train, predict_train, '-')
plt.show()


features.append('power_3')
polydata3 = polynomial_dataframe(train_data['sqft_living'], 3)
polydata3['price'] = y_train
X_train_3 = polydata3[features]
ml3 = LinearRegression()
ml3.fit(X_train_3, y_train)
predict_train = ml3.predict(X_train_3)
print(ml3.intercept_, ml3.coef_)
plt.plot(X_train, y_train, '.', X_train, predict_train, '-')
plt.show()


for i in range(4, 16):
  features.append('power_' + str(i))
polydata_15 = polynomial_dataframe(train_data['sqft_living'], 15)
polydata_15['price'] = y_train
X_train_15 = polydata_15[features]
ml_15 = LinearRegression()
ml_15.fit(X_train_15, y_train)
predict_train = ml_15.predict(X_train_15)
print(ml_15.intercept_, ml_15.coef_)
plt.plot(X_train, y_train, '.', X_train, predict_train, '-')
plt.show()


######################################################################################################################


def plot_model(data, features):
  new_data = polynomial_dataframe(data['sqft_living'], 15)
  X = new_data[features]
  y = data['price']
  Xs = data['sqft_living']
  model = LinearRegression()
  model.fit(X, y)
  predicted = model.predict(X)
  print(model.coef_)
  print(model.score(X, y))
  plt.plot(X['power_1'].reshape(len(X['power_1']), 1), y.reshape(len(y), 1), '.',
           X['power_1'].reshape(len(X['power_1']), 1), predicted, '-')
  plt.show()


def try_models(dtype_dict):
  data_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
  data_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
  data_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
  data_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
  features = get_features(15)
  plot_model(data_1, features)
  plot_model(data_2, features)
  plot_model(data_3, features)
  plot_model(data_4, features)


try_models(dtype_dict)


######################################################################################################################


def best_degree(train_data, cv_data, test_data):
  cv_Rss = []
  for i in range(1, 16):
    features = get_features(i)
    df_train = polynomial_dataframe(train_data['sqft_living'], i)
    df_cv = polynomial_dataframe(cv_data['sqft_living'], i)
    df_train['price'] = train_data['price']
    df_cv['price'] = cv_data['price']

    X_train = df_train[features]
    y_train = df_train['price']
    X_cv = df_cv[features]
    y_cv = df_cv['price']

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_cv)

    # plt.plot(cv_data['sqft_living'], y_cv, '.', cv_data['sqft_living'], predicted, '-')
    # plt.show()

    error = y_cv - predicted
    rss_error = (error**2).sum()
    print(i, rss_error)
    cv_Rss.append(rss_error)

  mn = min(cv_Rss)
  best_degree = 0
  for i in range(0, 15):
    if cv_Rss[i] == mn:
      best_degree = i + 1
      break

  features = get_features(best_degree)
  df_train = polynomial_dataframe(train_data['sqft_living'], best_degree)
  df_test = polynomial_dataframe(test_data['sqft_living'], best_degree)
  df_train['price'] = train_data['price']
  df_test['price'] = test_data['price']

  X_train = df_train[features]
  y_train = df_train['price']
  X_test = df_test[features]
  y_test = df_test['price']

  model = LinearRegression()
  model.fit(X_train, y_train)
  predicted = model.predict(X_test)
  error = y_test - predicted
  test_Rss = (error**2).sum()
  print('best degree : ', best_degree)
  print('test_Rss : ', test_Rss)


best_degree(train_data, cv_data, test_data)
