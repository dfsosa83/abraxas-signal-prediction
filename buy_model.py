# Standard library imports
import sys
import calendar
import pickle
from itertools import cycle, combinations

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sklearn
from sklearn import (
    model_selection, metrics, preprocessing, feature_selection, 
    ensemble, linear_model, svm, impute, cluster
)
import lightgbm as lgb
import xgboost as xgb
import torch
import tensorflow as tf
from keras.models import load_model
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.stats.diagnostic as smd
from scipy import stats
import ta

# Additional settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Read data
path = 'C:/Users/dsosa/Documents/abraxas-signal-prediction/data/processed/'
df = pd.read_csv(path + 'abrax_result_binary_buy_60m_v3_std.csv', delimiter=",",header=0,low_memory=False,float_precision='round_trip')

#convert datetime column to datetime
df['datetime'] = pd.to_datetime(df['datetime'])
#create 'year' column
df['Year'] = pd.to_datetime(df['datetime']).dt.year

#split data 
initial_start_date = datetime.strptime('2019-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
initial_end_date = datetime.strptime('2023-12-30 00:00:00', '%Y-%m-%d %H:%M:%S')

# Move the date range 7 days ahead
new_start_date = initial_start_date + timedelta(days=7)
new_end_date = initial_end_date + timedelta(days=7)

# Split the dataset using the new date range
initial_train_dataset_30m_0 = df[(df['datetime'] >= new_start_date) & (df['datetime'] <= new_end_date)]
initial_validation_dataset_30m = df[(df['datetime'] > new_end_date) & (df['datetime'] <= new_end_date + timedelta(days=152))]
non_seen_test_data_30m = df[df['datetime'] > new_end_date + timedelta(days=152)]
conformal_dataset_30m = df[df['datetime'] > new_end_date + timedelta(days=152)]

#print min max ranges for train and valid datasets
print('Initial Train Data Set:')
print(initial_train_dataset_30m_0['datetime'].min())
print(initial_train_dataset_30m_0['datetime'].max())

print('Validation Data Set:')
print(initial_validation_dataset_30m['datetime'].min())
print(initial_validation_dataset_30m['datetime'].max())

print('Test Data Set:')
print(non_seen_test_data_30m['datetime'].min())
print(non_seen_test_data_30m['datetime'].max())

print('Conformal Data Set:')
print(conformal_dataset_30m['datetime'].min())
print(conformal_dataset_30m['datetime'].max())

#data augmented
# Apply the augmentation function
from data_augmented import augment_datetime_months
augmented_df = augment_datetime_months(initial_train_dataset_30m_0, target_column='pip_class', specific_months=[9,10,11,12,1])

# Print some information about the original and augmented DataFrames
print(f"Original DataFrame shape: {initial_train_dataset_30m_0.shape}")
print(f"Augmented DataFrame shape: {augmented_df.shape}")

# Check the distribution of months in the augmented DataFrame
month_distribution = augmented_df['datetime'].dt.month.value_counts().sort_index()
print("\nMonth distribution in augmented DataFrame:")
print(month_distribution)

# Check the distribution of target variable in the augmented DataFrame
target_distribution = augmented_df['pip_class'].value_counts(normalize=True)
print("\nTarget variable distribution in augmented DataFrame:")
print(target_distribution)

#create initial_train_dataset_30m
initial_train_dataset_30m = augmented_df.copy()

#Modeling
#define training data set initial features and rename 'close'
initial_train_dataset_30m = initial_train_dataset_30m.rename(columns={'close_x': 'close'})
train_set = initial_train_dataset_30m[['datetime','open','high',	'low',	'close',	'tick_volume','spread','pip_class']]

#feature engineering
from feature_enginering import features_engineering
final_train_set = features_engineering(train_set)
print(final_train_set.shape)

#same process for validation non seen dataset and conformal data set
#date to datetime
initial_validation_dataset_30m['datetime'] = pd.to_datetime(initial_validation_dataset_30m['datetime'])
#rename validation set
modified_val = initial_validation_dataset_30m.copy()
#rename close_x
modified_val = modified_val.rename(columns={'close_x': 'close'})
#define validation dataset
validation_set = modified_val[['datetime','open','high',	'low',	'close',	'tick_volume','spread','pip_class']]
final_valid_set = features_engineering(modified_val)
print(final_valid_set.shape)
#non seen data
non_seen_test_data_30m['datetime'] = pd.to_datetime(non_seen_test_data_30m['datetime'])
#rename close_x
non_seen_test_data_30m = non_seen_test_data_30m.rename(columns={'close_x': 'close'})
# Create new features for the entire dataset
featured_data_test = features_engineering(non_seen_test_data_30m).copy()

#feature selection
from feature_selection import feature_selection
selected_features_names = feature_selection(final_train_set)
print("Selected Features:")
print(selected_features_names)

#create a new dataframe with columns_to_model and add 'DateTime' and 'class' features
df_to_model = final_train_set[list(selected_features_names) + ['datetime', 'pip_class']] # Convert selected_features_names to a list
print(df_to_model.shape)

#validation
df_to_model_val = final_valid_set[list(selected_features_names) + ['datetime', 'pip_class']] # Convert selected_features_names to a list
print(df_to_model_val.shape)

#non seen data
df_to_test = featured_data_test[list(selected_features_names) + ['datetime', 'pip_class']] # Convert selected_features_names to a list
print(df_to_test.shape)

#to model
train_dataset_ = df_to_model.copy()
test_dataset_ = df_to_model_val.copy()

##Model configuration
Y_train_cv, X_train_cv = train_dataset_.pip_class, train_dataset_
Y_validation, X_validation = test_dataset_.pip_class, test_dataset_

X_train_cv_final = X_train_cv[list(selected_features_names)]
X_validation_final = X_validation[list(selected_features_names)]

print(X_train_cv_final.shape)
print(X_validation_final.shape)

# Extract hour and month from datetime
X_train_cv['hour'] = pd.to_datetime(X_train_cv['datetime']).dt.hour
X_train_cv['month'] = pd.to_datetime(X_train_cv['datetime']).dt.month
X_validation['hour'] = pd.to_datetime(X_validation['datetime']).dt.hour
X_validation['month'] = pd.to_datetime(X_validation['datetime']).dt.month

# Define 'hot' hours and 'hot' months
hot_months = [9,10,11,12,1]  # Example hot months; adjust as needed

#define group
X_train_cv['group'] = X_train_cv.apply(lambda x: x['month'] if x['month'] in hot_months else -1, axis=1)
X_validation['group'] = X_validation.apply(lambda x: x['month'] if x['month'] in hot_months else -1, axis=1)
groups_train = X_train_cv['group']
groups_val = X_validation['group']

from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
gkf = GroupKFold(n_splits=5)

#train model
from model_training import train_lgbm_model
print('model training.........')
best_model, best_params, nested_cv_score, validation_predictions = train_lgbm_model(X_train_cv_final, Y_train_cv, X_validation_final, Y_validation, groups_train)
print(best_model.get_params())