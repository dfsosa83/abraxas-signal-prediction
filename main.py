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

# Local application imports
from data_processing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model

# Additional settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read data
path = 'C:/Users/dsosa/Documents/data/processed'

#data for buy
df_buy = pd.read_csv(path + '/abrax_result_binary_buy_60m_v3_std.csv')

#data for sell
df_sell = pd.read_csv(path + '/abrax_result_binary_sell_60m_v3_std.csv')



