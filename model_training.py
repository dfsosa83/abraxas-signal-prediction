from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score, GroupKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
from sklearn.metrics import precision_score, make_scorer, roc_auc_score, matthews_corrcoef

def train_lgbm_model(X_train_cv_final, Y_train_cv, X_validation_final, Y_validation, groups_train, class_weight={0: 1, 1: 2.3}, random_state=314, n_iter=50):
    # Initialize a LGBMClassifier with class weights
    clf = lgb.LGBMClassifier(class_weight=class_weight)

    # Set up possible parameters to randomly sample from
    param_test = {
        'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05],
        'n_estimators': sp_randint(100, 250),
        'max_depth': sp_randint(8, 12),
        'num_leaves': sp_randint(5, 31),
        'min_child_samples': sp_randint(10, 50),
        'min_child_weight': [0.001, 0.01, 0.1],
        'subsample': sp_uniform(loc=0.6, scale=0.4),
        'colsample_bytree': sp_uniform(loc=0.6, scale=0.4),
        'reg_alpha': [0],
        'reg_lambda': [0],
        'min_split_gain': [0],
        'boosting_type': ['gbdt', 'dart'],
        'force_col_wise': [True],
        'max_bin': sp_randint(200, 300)
    }

    # A dictionary of the metrics of interest
    scoring = {
        'mcc': make_scorer(matthews_corrcoef),
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall'
    }

    # Create the GroupKFold object
    gkf = GroupKFold(n_splits=5)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(clf, param_distributions=param_test,
                                       n_iter=n_iter, scoring=scoring,
                                       cv=gkf.split(X_train_cv_final, Y_train_cv, groups=groups_train),
                                       refit='roc_auc',
                                       random_state=random_state, verbose=True)

    # Fit randomized_search
    random_search.fit(X_train_cv_final, Y_train_cv,
                      eval_set=[(X_validation_final, Y_validation)],
                      eval_metric='auc',
                      groups=groups_train)

    # Get the best parameters from the RandomizedSearchCV
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # Create a new classifier using the best parameters from the RandomizedSearchCV
    clf_best = lgb.LGBMClassifier(**best_params, class_weight=class_weight)

    # Nested CV with parameter optimization using GroupKFold
    outer_cv = GroupKFold(n_splits=5)
    nested_score = cross_val_score(clf_best, X=X_train_cv_final, y=Y_train_cv,
                                   groups=groups_train, cv=outer_cv, scoring='roc_auc')
    print("Nested CV score: ", nested_score.mean())

    # Train a new model on the whole training set using the best parameters found
    clf_best.fit(X_train_cv_final, Y_train_cv,
                 eval_set=[(X_validation_final, Y_validation)],
                 eval_metric='auc')

    # Test the trained model on the validation set
    predictions = clf_best.predict(X_validation_final)

    return clf_best, best_params, nested_score.mean(), predictions


