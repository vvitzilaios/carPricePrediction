from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(input_path):
    data = pd.read_csv(input_path)
    return data


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


def preprocess_features(data):
    # Derive 'Car Age'
    data["Car Age"] = data["Prod. year"].apply(lambda x: datetime.now().year - x)

    # Drop unnecessary columns
    columns_to_drop = ["Prod. year", "Doors", "Cylinders", "ID", "Wheel"]
    data.drop(columns=columns_to_drop, inplace=True)

    # Handle Levy
    data["Levy"].replace("-", 0, inplace=True)
    data["Levy"] = data["Levy"].astype(int)

    # Handle 'Engine volume'
    data["Turbo"] = data["Engine volume"].apply(lambda x: 1 if "Turbo" in x else 0)
    data["Engine volume"] = (
        data["Engine volume"].str.replace(" Turbo", "").astype(float)
    )

    # Convert 'Mileage'
    data["Mileage"] = data["Mileage"].str.replace(" km", "").astype(int)

    return data


def encode_and_scale_features(data):
    categorical_cols = [
        "Manufacturer",
        "Model",
        "Category",
        "Leather interior",
        "Fuel type",
        "Gear box type",
        "Drive wheels",
        "Color",
    ]
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Scale continuous features
    scaler = StandardScaler()
    continuous_features = ["Levy", "Engine volume", "Mileage", "Airbags", "Car Age"]
    data_encoded[continuous_features] = scaler.fit_transform(
        data_encoded[continuous_features]
    )

    return data_encoded, scaler


def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering out the outliers
    data_outliers_removed = data[
        (data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)
    ]

    return data_outliers_removed


def preprocess_data(input_path, output_path):
    data = load_data(input_path)
    print(data.head())

    # Plotting
    plot_categorical_counts(
        data, "Manufacturer", save_path="../plots/car_number_per_manufacturer.png"
    )
    plot_avg_price_per_manufacturer(
        data, save_path="../plots/average_price_per_manufacturer.png"
    )

    # Preprocessing
    data = preprocess_features(data)
    data = remove_outliers(data, "Price")
    data_encoded, scaler = encode_and_scale_features(data)

    # Check NaN values
    if data_encoded.isnull().sum().sum() > 0:
        print("Warning: NaN values detected!")
        print(data_encoded.isnull().sum())

    # Save processed data
    data_encoded.to_csv(output_path, index=False)

    return data_encoded, scaler


if __name__ == "__main__":
    input_file_path = "../data/car_price_prediction.csv"
    output_file_path = "../data/car_price_prediction_processed.csv"
    preprocess_data(input_file_path, output_file_path)
