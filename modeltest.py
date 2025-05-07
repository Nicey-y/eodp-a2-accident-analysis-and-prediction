import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# print the table that we want to a .csv file
def custom_compare_models(df):
    # Evaluate if the data is balanced or imbalanced

    # Scaling?
    
    train_X = df.drop("SEVERITY", axis=1) # !reminder, classification and regression models take different SEVERITY colums
    train_Y = df["SEVERITY"].copy()

    # Divide into variations !!!!!!!!!!!!!! CHANGE IN THE FUTUTURE BASED ON CORRELATION ANALYSIS !!!!!!!!!!!!!!!
    external_cols = ['SURFACE_COND', 'ATMOSPH_COND', 'LIGHT_CONDITION', 'SPEED_ZONE', 'ROAD_GEOMETRY']
    internal_cols = ['SEATING_POSITION', 'HELMENT_BELT_WORN', 'LICENCE_STATE']
    all_cols = list(train_X.columns)
    
    variations = {
        0: {
            "name": "Prediction based on external factors",
            "columns": external_cols
        },
        1: {
            "name": "Prediction based on internal factors",
            "columns": internal_cols
        },
        2: {
            "name": "Prediction based on both external and internal factors",
            "columns": all_cols
        }
    }
    

    ######### REGRESSON MODELS #########
    # Models Initialisation
    lin_reg = LinearRegression()
    tree_reg = DecisionTreeRegressor()
    forest_reg = RandomForestRegressor()
    
    # Hyperparameters for fine tuning
    models_configs = {
        0 : {
            "name": 'Linear Regressor',
            "model": lin_reg,
            "param_grid": {
                "fit_intercept": [True],
                "copy_X": [True]
                }
        },
        1 : {
            "name": "Decision Tree Regressor",
            "model": tree_reg,
            "param_grid": {
                "criterion": ["gini", "entropy"], # ??????????
                "max_depth": [3, None],
                "min_samples_leaf": randint(1, 9),
                "max_features": randint(1, 9)
                }
        },
        2 : {
            "name": "Random Forest Regressor",
            "model": forest_reg,
            "param_grid": {
                'n_estimators': [130, 180, 230],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt'],
                'bootstrap': [True, False]
                }
        }
    }
    
    # scorings = {'accuracy': make_scorer(accuracy_score),
    #            'prec': 'precision'}

    # Evaluation
    # for each variation
    for i in range(0, 3):
        print(variations[i]['name'])
        X_cols = train_X.copy()[variations[i]['columns']]
        # for each model
        for j in range(0, 3):
            # Fine tuning
            rand_search = RandomizedSearchCV(models_configs[j]["model"], models_configs[j]["param_grid"], cv=10,
                                             scoring='neg_mean_squared_error', # forcing to negative to make it easier to sqrt
                                             return_train_score=True)
            rand_search.fit(X_cols, train_Y)

            # print (hyperparameters, RMSE) of the best model
            print(models_configs[j]["name"])
            eval_res = pd.DataFrame(rand_search.cv_results_)
            param_res = pd.DataFrame(rand_search.best_params_)
            res_df = pd.concat([eval_res, param_res])
            print(res_df.head())
            # for mean_score, params in zip(eval_res["mean_test_score"], eval_res["params"]):
            #     print(np.sqrt(-mean_score), params) # printing for now, change to writing to csv later

# Takes a dataframe with all x values, but y values with labels only
def compare_classification_models(df, variations):
    # Evaluate if the data is balanced or imbalanced
    # Scaling?
    
    # Splitting into train set and test set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Provides train/test indices to split data in train/test sets.
    for train_index, test_index in split.split(df, df["SEVERITY_LABEL"]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    train_X = train_set.drop("SEVERITY_LABEL", axis=1) # not sure if i have to drop them or not, will inspect once i have data on hands
    train_Y = train_set["SEVERITY_LABEL"].copy()
    test_X = test_set.drop("SEVERITY_LABEL", axis=1)
    test_Y = test_set["SEVERITY_LABEL"].copy()

    headers = ['k', 'Independent Variable Type', 'Recall', 'Precision', 'F1-Score']
    
    ######### K-Nearest Neighbours #########
    knn = KNeighborsClassifier()

    # Fine tune the n_neighbors=k hyperparameter    
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        # for each variation
        for i in range(0, 3):
            print(variations[i]['name'])
            X_cols = train_X.copy()[variations[i]['columns']]
            knn.fit(X_cols, train_Y)
            pred_y = knn.predict(test_X)
            recall = round(recall_score(test_Y, pred_y, average='weighted'),2)
            precidion = round(precision_score(test_Y, pred_y, average='weighted'),2)
            f1 = round(f1_score(test_Y, pred_y,average='weighted'),2)

    # if accuracy varies a lot between different n_neighbors=k values then the model is not robust

    ######### Decision Tree #########
    