import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_counts(data, column_name, save_path=None):
    plt.figure(figsize=(20, 10))
    plt.bar(data[column_name].value_counts().index, data[column_name].value_counts())
    plt.xticks(rotation=90)
    plt.xlabel(column_name)
    plt.ylabel("Counts")
    plt.title(f"Number of cars per {column_name}")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_avg_price_per_manufacturer(data, save_path=None):
    avg_prices = data.groupby("Manufacturer")["Price"].mean()
    plt.figure(figsize=(20, 10))
    plt.bar(avg_prices.index, avg_prices)
    plt.xticks(rotation=90)
    plt.xlabel("Manufacturer")
    plt.ylabel("Average Price")
    plt.title("Average price per manufacturer")

    if save_path:
        plt.savefig(save_path)
    plt.show()


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


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title(f"{model_name} Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"../plots/{model_name.replace(' ', '_')}_residuals.png")
    plt.show()


def residuals_distribution(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} Distribution of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"../plots/{model_name.replace(' ', '_')}_residuals_distribution.png")
    plt.show()


def plot_aggregated_importance(aggregated):
    # Sort features based on importance
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

    # Unzip the sorted items into two lists
    labels, values = zip(*sorted_features)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Importance")
    plt.title("Aggregated Feature Importances")
    plt.gca().invert_yaxis()  # To display the most important feature at the top

    plt.savefig(f"../plots/aggregated_feature_importance.png")
    plt.show()


def aggregate_importance(importance, columns):
    # Create a dictionary to store the aggregated importance
    aggregated = {}

    for col, imp in zip(columns, importance):
        # For each column, find the original feature name (before one-hot encoding)
        # Assuming column names of dummies are like 'FeatureName_Value'
        original_feature = col.split("_")[0]

        if original_feature in aggregated:
            aggregated[original_feature] += imp
        else:
            aggregated[original_feature] = imp

    return aggregated
