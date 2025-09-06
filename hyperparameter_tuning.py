"""
hyperparameter_running.py
-------------------------
Tests different hyperparameters for perceptron.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron

def run_hyperparameter_search(X_train, y_train):
    """
    Runs a simple hyperparameter search with GridSearchCV.
    """
    param_grid = {
        'max_iter': [500, 1000, 2000],
        'eta0': [0.1, 1.0, 10.0]
    }
    model = Perceptron(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_
