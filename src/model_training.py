"""
Módulo para treinamento e avaliação de modelos.
"""
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=376):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train, kf, param_grid):
    pipe = Pipeline([
        ('clf', LogisticRegression(max_iter=1000, random_state=376))
    ])
    grid = GridSearchCV(pipe, param_grid, cv=kf, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid

def train_random_forest(X_train, y_train, kf, param_grid):
    pipe = Pipeline([
        ('clf', RandomForestClassifier(random_state=376))
    ])
    grid = GridSearchCV(pipe, param_grid, cv=kf, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
