from timeit import default_timer as timer

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split, GridSearchCV

import util.models as models
from util.plots import (
    plot_predictions,
    aggregate_importance,
    plot_aggregated_importance,
)
from util.preprocess import preprocess_data


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

    models_dict = models.get_models()
    search_spaces = models.get_search_spaces()
    total_time = 0
    for name, model in models_dict.items():
        start_time_model = timer()
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

        end_time_model = timer()
        total_model_time = end_time_model - start_time_model
        total_time += total_model_time

        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(
            f"{name} RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f} %, R2 Score: {r2:.3f},"
            f" Time elapsed: {total_model_time:.3f}s"
        )
        plot_predictions(y_test, predictions, name)
        if name == "Random Forest":
            importance = best_model.feature_importances_
            aggregated = aggregate_importance(importance, X_train.columns)
            plot_aggregated_importance(aggregated)

    print(f"Total time elapsed: {total_time:.3f}s")


if __name__ == "__main__":
    train_models()
