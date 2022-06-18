import os
import pickle
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit


def run_benchmarks(path_to_feature_dict):
    """
        Assumes data is already split into train/test/val (the same split as for training deep models)
        Implements Gridsearch for classifier params
        Uses cross validation, with indices of pre-existing train and val sets

    """

    #TODO - seed

    with open(path_to_feature_dict, 'rb') as f:
        feature_dict = pickle.load(f)
    print(feature_dict.keys() )

    assert 'train' in feature_dict.keys() and 'test'in feature_dict.keys() and 'val' in feature_dict.keys() 
    train_df = feature_dict["train"]
    test_df = feature_dict["test"]
    val_df = feature_dict["val"]
    X_train = train_df.iloc[:, :-1]
    y_train = train_df['label']
    X_test = test_df.iloc[:, :-1]
    y_test = test_df['label']
    X_val = val_df.iloc[:, :-1]
    y_val = val_df['label']
    

    merged_X = pd.concat((X_train, X_val), axis=0).to_numpy()
    merged_y = pd.concat((y_train, y_val), axis=0).to_numpy()
    split = [(range(len(X_train)), range(
        len(X_train), len(X_train) + len(X_val)))]
    print(split)
    print(merged_X.shape)

    mytestfold = []

    for i in range(len(X_train)):
        mytestfold.append(-1)
    for i in range(len(X_val)):
        mytestfold.append(0)

    split = PredefinedSplit(test_fold=mytestfold)

    clf = RandomForestClassifier()
    model = Pipeline([
        ('sampling', SMOTE()),
        ('randomforestclassifier', clf)
    ])

    param_grid = [{
        'randomforestclassifier__max_depth': [2, 3, 4],
        'randomforestclassifier__max_features':[2, 3, 4, 5, 6]
    }]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=split)
    grid.fit(merged_X, merged_y)
    y_pred = grid.predict(X_test)
    print(y_pred)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(clf , " accuracy:", accuracy)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

    clf = KNeighborsClassifier()
    model = Pipeline([
        ('sampling', SMOTE()),
        ('kneighborsclassifier', clf)
    ])

    param_grid = [{
        'kneighborsclassifier__n_neighbors': (1, 10, 1),
        'kneighborsclassifier__leaf_size': (20, 40, 1),
        'kneighborsclassifier__p': (1, 2),
        'kneighborsclassifier__weights': ('uniform', 'distance'),
        'kneighborsclassifier__metric': ('minkowski', 'chebyshev')}]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=split)
    grid.fit(merged_X, merged_y)
    y_pred = grid.predict(X_test)
    print(y_pred)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(clf ,"accuracy:", accuracy)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

    clf = tree.DecisionTreeClassifier()
    model = Pipeline([
        ('sampling', SMOTE()),
        ('dtclassifier', clf)
    ])

    param_grid = [{'dtclassifier__criterion': ['entropy', 'gini'],
                   'dtclassifier__max_depth': [2, 3, 4]
                }]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=split)
    grid.fit(merged_X, merged_y)
    y_pred = grid.predict(X_test)
    print(y_pred)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(clf ,"accuracy:", accuracy)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

    return