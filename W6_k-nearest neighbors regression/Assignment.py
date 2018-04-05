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


def get_numpy_data(data_frame, features, output):
  data_frame['constant'] = 1
  features = ['constant'] + features

  all_features_2d_array = data_frame[features]
  all_features_2d_array_to_numpy = np.array(all_features_2d_array)

  output_array = data_frame[output]
  output_array_numpy = np.array(output_array)
  return (all_features_2d_array_to_numpy, output_array_numpy)


def normalize_features(features):
  norms = np.linalg.norm(features, axis=0)
  normalized_features = features / norms
  return (normalized_features, norms)


def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2)**2))


def compute_distances(features_instances, features_query):
  diff = features_query - features_instances
  dists = np.sqrt(np.sum(diff**2, axis=1))
  return dists


def compute_k_nearest_neighbors(k, features_matrix, feature_vector):
  dists = compute_distances(features_matrix, feature_vector)
  return np.argsort(dists, axis=0)[:k]


def compute_distances_k_avg(k, features_matrix, output_values, feature_vector):
  k_neigbors = compute_k_nearest_neighbors(k, features_matrix, feature_vector)
  avg_value = np.mean(output_values[k_neigbors])
  return avg_value


def compute_distances_k_all(k, features_matrix, output_values, feature_vector):
  num_of_rows = feature_vector.shape[0]
  predicted_values = []
  for i in xrange(num_of_rows):
    avg_value = compute_distances_k_avg(k, features_matrix, output_values, feature_vector[i])
    predicted_values.append(avg_value)
  return predicted_values


# Choosing the best value of k using a validation set
def compute_best_k(K, training_X, training_y, validation_X, validation_y):
  rss_vals = []
  kvals = range(1, K)
  for k in range(1, K):
    predicted = compute_distances_k_all(k, training_X, training_y, validation_X)
    error = predicted - validation_y
    rss = (error**2).sum()
    rss_vals.append(rss)
  best_k = rss_vals.index(min(rss_vals)) + 1

  plt.plot(kvals, rss_vals, 'bo-')
  plt.show()
  return best_k

##########################################################################################################################


dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('kc_house_data_small.csv', dtype=dtype_dict)
testing = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)
training = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
validation = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)


features = ['bedrooms',
            'bathrooms',
            'sqft_living',
            'sqft_lot',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15']

output = ['price']
training_X, training_y = get_numpy_data(training, features, output)
validation_X, validation_y = get_numpy_data(validation, features, output)
testing_X, testing_y = get_numpy_data(testing, features, output)

training_X, training_norms = normalize_features(training_X)
validation_X = validation_X / training_norms
testing_X = testing_X / training_norms

##########################################################################################################################

dict_dist = {}
for i in range(0, 10):
  dict_dist[i] = euclidean_distance(testing_X[0], training_X[i])
print(dict_dist)

distance = []
for x, y in dict_dist.items():
  distance.append((y, x))
distance.sort()
print(distance)


dists = compute_distances(training_X, testing_X[2])
min_index = dists.argsort()[0]
print(training_y[min_index])
print(min_index)


print compute_k_nearest_neighbors(4, training_X, testing_X[2])
print compute_distances_k_avg(4, training_X, training_y, testing_X[2])

predicted_values = compute_distances_k_all(10, training_X, training_y, testing_X[0:10])
print predicted_values
print predicted_values.index(min(predicted_values))


best_k = compute_best_k(16, training_X, training_y, validation_X, validation_y)

# rss on test_data with best k
predicted = compute_distances_k_all(best_k, training_X, training_y, testing_X)
error = predicted - testing_y
rss = (error**2).sum()
print(rss)
