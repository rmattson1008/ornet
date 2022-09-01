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
from sklearn.decomposition import PCA, KernelPCA
import seaborn




def run_benchmarks(feature_dict):
    assert 'train' in feature_dict.keys() and 'test'in feature_dict.keys() and 'val' in feature_dict.keys() 
    # train_df = feature_dict["train"].copy()
    # test_df = feature_dict["test"].copy()
    # val_df = feature_dict["val"].copy()

    show_pca(feature_dict)

    print("Benchmarks for MDIVI vs LLO")
    train_df = feature_dict["train"].copy()
    train_df  = train_df.loc[train_df["label"] != 0]
    test_df = feature_dict["test"].copy()
    test_df  = test_df.loc[test_df["label"] != 0]
    val_df = feature_dict["val"].copy()
    val_df  = val_df.loc[val_df["label"] != 0]
    run_models(train_df, test_df, val_df)

    print("Benchmarks for Control vs LLO")
    train_df = feature_dict["train"].copy()
    train_df  = train_df.loc[train_df["label"] != 1]
    test_df = feature_dict["test"].copy()
    test_df  = test_df.loc[test_df["label"] != 1]
    val_df = feature_dict["val"].copy()
    val_df  = val_df.loc[val_df["label"] != 1]
    run_models(train_df, test_df, val_df)
    

    print("Benchmarks for MDIVI vs Control")
    train_df = feature_dict["train"].copy()
    train_df  = train_df.loc[train_df["label"] != 2]
    test_df = feature_dict["test"].copy()
    test_df  = test_df.loc[test_df["label"] != 2]
    val_df = feature_dict["val"].copy()
    val_df  = val_df.loc[val_df["label"] != 2]
    run_models(train_df, test_df, val_df)




def run_models(train_df, test_df, val_df):
    """
        Assumes data is already split into train/test/val (the same split as for training deep models)
        Implements Gridsearch for classifier params
        Uses cross validation, with indices of pre-existing train and val sets

    """

  
    
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
    # print(split)
    # print(merged_X.shape)

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
    # print(y_pred)
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
    # print(y_pred)
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
    # print(y_pred)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(clf ,"accuracy:", accuracy)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

    return


def show_pca(feature_dict):
    train_df = feature_dict["train"]
    test_df = feature_dict["test"]
    val_df = feature_dict["val"]

    X_train = train_df.iloc[:, :-1]
    y_train = train_df['label']
    X_test = test_df.iloc[:, :-1]
    y_test = test_df['label']
    # X_val = val_df.iloc[:, :-1]
    # y_val = val_df['label']

    pca = PCA(n_components=2)

    X_test_pca = pca.fit(X_train).transform(X_test)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
    plt.show()


def show_conf_mat(conf_mat, class_label_list, title=''):
    num_classes = len(class_label_list)
    ticks = np.arange(num_classes)+0.5
    conf_mat = conf_mat / np.sum(conf_mat)
    print(conf_mat)
    seaborn.heatmap(conf_mat, cmap='RdPu', 
            xticklabels=class_label_list, yticklabels=class_label_list,
            annot=True, annot_kws={'size': 12, 'weight':'bold'})
    
    plt.title(title, fontsize=18); 
    plt.xlabel('Prediction', fontsize=16);
    plt.xlim(0,num_classes)
    plt.xticks(ticks, class_label_list, rotation=45)
    
    plt.ylabel('Truth', fontsize=16);
    plt.ylim(0,num_classes)
    plt.yticks(ticks, class_label_list,rotation='horizontal') 
#     plt.invert_yaxis();
    plt.gca().invert_yaxis()
    plt.axis('auto')
    plt.show()