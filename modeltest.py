import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import csv

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV

# print the table that we want to a .csv file
def custom_compare_models(df):
        
    cpy = df.copy().drop("SEVERITY", axis=1) # remove SEVERITY column
    cpy = cpy.drop('ACCIDENT_NO', axis=1) # remove ACCIDENT_NO column
    # Divide into two types of factors
    external_cols = ['SPEED_ZONE','ROAD_GEOMETRY','NO_PERSONS_NOT_INJ','TAKEN_HOSPITAL','MEDIAN_AGE_GROUP','DAY_OF_WEEK','NO_OF_VEHICLES','AGG_LIGH_SURF_ATMOS_COND']
    internal_cols = ['LICENCE_STATE','SEATING_POSITION','HELMET_BELT_WORN','NO_PERSONS']
    all_cols = list(cpy.columns)

    variations = {
        0: {
            "name": "external",
            "columns": external_cols
        },
        1: {
            "name": "internal",
            "columns": internal_cols
        },
        2: {
            "name": "all",
            "columns": all_cols
        }
    }

    compare_classification_models(df, variations)


# Takes a dataframe with all x values, but y values with labels only
# Print comparison to a csv file
def compare_classification_models(df, variations):

    # Evaluate if the data is balanced or imbalanced
    balanced = is_balanced(df["SEVERITY"])
    # balanced = True 
    
    # Splitting into train set and test set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Provides train/test indices to split data in train/test sets.
    for train_index, test_index in split.split(df, df["SEVERITY"]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    train_X = train_set.drop("SEVERITY", axis=1)
    train_Y = train_set["SEVERITY"].copy()
    test_X = test_set.drop("SEVERITY", axis=1)
    test_Y = test_set["SEVERITY"].copy()

    # K-Nearest Neighbours Classification
    # compare_knn(train_X, train_Y, test_X, test_Y, variations, balanced)

    # Decision Tree Classification
    compare_dt(train_X, train_Y, test_X, test_Y, variations, balanced)


# K-Nearest Neighbours
def compare_knn(train_X, train_Y, test_X, test_Y, variations, balanced):
    file_path = 'knn.csv'
    knn_balanced_headers = ['k', 'Independent Variable Type', 'Accuracy']
    knn_imbalanced_headers = ['k', 'Independent Variable Type', 'Recall', 'Precision', 'F1-Score']
    
    # Write headers to csv
    with open(file_path, 'w') as file:
        file.write("K-Nearest Neighbours\n")
        if balanced:
            file.write(','.join(knn_balanced_headers))
        else:
            file.write(','.join(knn_imbalanced_headers))
        file.write("\n")

    knn = KNeighborsClassifier()
    # Fine tune the n_neighbors=k hyperparameter    
    for k in range(5, 16):
        knn = KNeighborsClassifier(n_neighbors=k)
        # for each variation
        for i in range(0, 3):
            print(variations[i]['name']) # debug
            X_cols = train_X.copy()[variations[i]['columns']]
            knn.fit(X_cols, train_Y)
            pred_y_knn = knn.predict(test_X.copy()[variations[i]['columns']])
            if balanced:
                accuracy_knn = balanced_evaluate(test_Y, pred_y_knn)
                
                # write to csv
                with open(file_path, 'a') as file:
                    file.write(','.join([str(k), variations[i]['name'], str(accuracy_knn)]))
                    file.write("\n")
            else:
                recall_knn, precision_knn, f1_knn = imbalanced_evaluate(test_Y, pred_y_knn)
                
                # write to csv
                with open(file_path, 'a') as file:
                    file.write(','.join([str(k), variations[i]['name'], str(recall_knn), str(precision_knn), str(f1_knn)]))
                    file.write("\n")

    # if accuracy varies a lot between different n_neighbors=k values then the model is not robust

def compare_dt(train_X, train_Y, test_X, test_Y, variations, balanced):
    file_path = 'dt.csv'
    dt_balanced_headers = ['Independent Variable Type', 'Accuracy']
    dt_imbalanced_headers = ['Independent Variable Type', 'Recall', 'Precision', 'F1-Score']

    # Write to csv
    with open(file_path, 'w') as file:
        file.write("Decision Tree Classification\n")
        if balanced:
            file.write(','.join(dt_balanced_headers))
        else:
            file.write(','.join(dt_imbalanced_headers))
        file.write("\n")

    train_Y.to_numpy().reshape(-1,1)
    dt = DecisionTreeClassifier(criterion='entropy')  # we specify entropy for IG
    # for each variation
    for i in range(0, 3):
        print(variations[i]['name'])
        X_cols = train_X.copy()[variations[i]['columns']]
        dt.fit(X_cols, train_Y)
        pred_y_dt = dt.predict(test_X.copy()[variations[i]['columns']])
        if balanced:
            accuracy_dt = balanced_evaluate(test_Y, pred_y_dt)

            # write to csv
            with open(file_path, 'a') as file:
                file.write(','.join([variations[i]['name'], str(accuracy_dt)]))
                file.write("\n")
        else:
            recall_dt, precision_dt, f1_dt = imbalanced_evaluate(test_Y, pred_y_dt)

            # write to csv
            with open(file_path, 'a') as file:
                file.write(','.join([variations[i]['name'], str(recall_dt), str(precision_dt), str(f1_dt)]))
                file.write("\n")

def is_balanced(col):
    max_diff = 0.05
    freq = col.value_counts(normalize=True) # normalize=True to get the frequency
    for indx1 in freq.index:
        for indx2 in freq.index:
            diff = abs(freq[indx1] - freq[indx2])
            if diff > max_diff:
                return False
    return True

def balanced_evaluate(test_Y, pred_y):
    return round(accuracy_score(test_Y, pred_y), 2)

def imbalanced_evaluate(test_Y, pred_y):
    recall_dt = round(recall_score(test_Y, pred_y, average='weighted', zero_division=1.0),2)
    precidion_dt = round(precision_score(test_Y, pred_y, average='weighted', zero_division=1.0),2)
    f1_dt = round(f1_score(test_Y, pred_y,average='weighted'),2)
    return recall_dt, precidion_dt, f1_dt

df = pd.read_csv('accident_processed_new(3).csv')
custom_compare_models(df)