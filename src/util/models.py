from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


def get_models():
    return {
        "SGD Regression": SGDRegressor(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "kNN": KNeighborsRegressor(),
    }


def get_search_spaces():
    sgd_params = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "loss": ["squared_error"],
        "penalty": [None],
        "max_iter": [10000],
        "tol": [1e-3],
    }

    ridge_params = {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}

    rf_params = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    return {
        "SGD Regression": sgd_params,
        "Ridge Regression": ridge_params,
        "Random Forest": rf_params,
        "kNN": knn_params,
    }


def train_best_model(model, search_space, X_train, y_train):
    if search_space:
        search = GridSearchCV(model, search_space, cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model
