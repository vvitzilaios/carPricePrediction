from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from util.preprocess import preprocess_data


def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        np.linspace(min(y_true), max(y_true), 100),
        np.linspace(min(y_true), max(y_true), 100),
        color="red",
        linestyle="--",
    )

    plt.title(f"{model_name} Predictions vs Actual")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"../plots/{model_name.replace(' ', '_')}_predictions_vs_actual.png")
    plt.show()


def train_models():
    input_file_path = "../data/car_price_prediction.csv"
    output_file_path = "../data/car_price_prediction_processed.csv"
    data, scaler = preprocess_data(input_file_path, output_file_path)

    X = data.drop(columns=["Price"])
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    continuous_features = ["Levy", "Engine volume", "Mileage", "Airbags", "Car Age"]
    X_train[continuous_features] = scaler.transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "kNN": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }

    # Hyperparameter grid for Ridge Regression
    ridge_params = {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}

    # Hyperparameter grid for Random Forest
    rf_params = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Hyperparameter grid for kNN
    knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    # Hyperparameter grid for Gradient Boosting
    gb_params = {
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Search spaces
    search_spaces = {
        "Linear Regression": {},
        "Ridge Regression": ridge_params,
        "Random Forest": rf_params,
        "kNN": knn_params,
        "Gradient Boosting": gb_params,
    }

    for name, model in models.items():
        # Check if there are hyperparameters to tune
        if search_spaces[name]:
            # Hyperparameter tuning using GridSearchCV
            search = GridSearchCV(model, search_spaces[name], cv=3, n_jobs=-1)
            search.fit(X_train, y_train)

            # Use the best estimator
            best_model = search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        predictions = best_model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        print(f"{name} RMSE: {rmse} R2 Score: {r2}")
        plot_predictions(y_test, predictions, name)


if __name__ == "__main__":
    train_models()
