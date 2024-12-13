import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb

def feature_selection(final_train_set, top_n_features=30, vote_threshold=2, random_state=42):
    # Number of samples should match the number of rows in your dataset
    data_size = final_train_set.shape[0]
    
    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Generate Gaussian noise variables with different standard deviations
    noise_gaussian_1 = np.random.normal(loc=0, scale=1, size=data_size)  # Std dev = 1

    # Generate Uniform noise variables within a specified range
    noise_uniform_1 = np.random.uniform(low=-1, high=1, size=data_size)  # Uniform between -1 and 1

    # Generate Poisson noise variables with a specified lambda value
    lambda_value = 3  # Mean number of occurrences
    poisson_noise_1 = np.random.poisson(lam=lambda_value, size=data_size)
    poisson_noise_3 = np.random.poisson(lam=lambda_value * 3, size=data_size)  # Even higher lambda

    # Add the noise variables to the final_train_set DataFrame
    final_train_set['Gaussian_Noise_Std1'] = noise_gaussian_1
    final_train_set['Uniform_Noise_Range1'] = noise_uniform_1
    final_train_set['Poisson_Noise_Lambda1'] = poisson_noise_1
    final_train_set['Poisson_Noise_Lambda3'] = poisson_noise_3

    # Define the features (X) and target variable (y)
    X = final_train_set.drop(['pip_class','datetime'], axis=1)
    y = final_train_set['pip_class']

    # Initialize the models
    model1 = RandomForestClassifier(random_state=random_state)
    model2 = lgb.LGBMClassifier(random_state=random_state)
    model3 = RidgeClassifier(random_state=random_state)

    # Fit the models
    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)

    # Select top features using feature importances or coefficients
    selector1 = SelectFromModel(model1, prefit=True, max_features=top_n_features)
    selector2 = SelectFromModel(model2, prefit=True, max_features=top_n_features)
    selector3 = SelectFromModel(model3, prefit=True, max_features=top_n_features)

    # Get the selected features
    selected_features1 = selector1.get_support()
    selected_features2 = selector2.get_support()
    selected_features3 = selector3.get_support()

    # Count the votes for each feature
    feature_votes = selected_features1.astype(int) + selected_features2.astype(int) + selected_features3.astype(int)

    # Select features with votes above the threshold
    selected_features = feature_votes >= vote_threshold

    # Get the selected feature names
    selected_features_names = X.columns[selected_features]

    return selected_features_names

