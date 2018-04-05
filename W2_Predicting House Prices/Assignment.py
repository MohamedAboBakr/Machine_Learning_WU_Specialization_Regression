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


def add_features(df):
  df['bedrooms_squared'] = df['bedrooms'] * df['bedrooms']
  df['bed_bath_rooms'] = df['bedrooms'] * df['bathrooms']
  df['log_sqft_living'] = np.log(df['sqft_living'])
  df['lat_plus_long'] = df['lat'] + df['long']


train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')

add_features(train_data)
add_features(test_data)

y_train = train_data['price']
y_test = test_data['price']


features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
X1_train = train_data[features]
X1_test = test_data[features]
ml1 = LinearRegression()
ml1.fit(X1_train, y_train)
coef1 = ml1.coef_
accuracy1 = ml1.score(X1_test, y_test)
print(coef1)
print(accuracy1)


features.append('bed_bath_rooms')
X2_train = train_data[features]
X2_test = test_data[features]
ml2 = LinearRegression()
ml2.fit(X2_train, y_train)
coef2 = ml2.coef_
accuracy2 = ml2.score(X2_test, y_test)
print(coef2)
print(accuracy2)


features.append('bedrooms_squared')
features.append('log_sqft_living')
features.append('lat_plus_long')
X3_train = train_data[features]
X3_test = test_data[features]
ml3 = LinearRegression()
ml3.fit(X3_train, y_train)
coef3 = ml3.coef_
accuracy3 = ml3.score(X3_test, y_test)
print(coef3)
print(accuracy3)
