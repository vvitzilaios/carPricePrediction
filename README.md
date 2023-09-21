# Car Price Prediction Application

Predict the price of a car using various regression models.

## Overview

This application utilizes multiple regression models to predict car prices based on a dataset containing various features of cars. Different models including SGD Regression, Ridge Regression, K-Nearest Neighbors, and Random Forest are trained and evaluated to determine the best performing model.

## Features

1. **Data Preprocessing**: A preprocessing pipeline that transforms the raw dataset, handling missing values, outliers, and scaling.
2. **Exploratory Data Analysis (EDA)**: Visualizations to provide insights on the dataset, such as distribution of prices, number of cars by manufacturer, and average price by manufacturer.
3. **Model Training and Evaluation**: Implementations for training multiple models and evaluating their performances using metrics such as RMSE, MAE, MAPE, and R2 Score.
4. **Visual Analysis**: Functions to plot predictions vs. actual values, residuals distribution, and feature importances.

## Getting Started

1. Clone the repository:  
   `git clone https://github.com/your-username/car-price-prediction.git`
2. Navigate to the directory:  
   `cd car-price-prediction`
3. Install required libraries (preferably in a virtual environment):  
   `pip install -r requirements.txt`
4. Launch the Jupyter notebook or Google Colab to interact with the provided notebooks.
5. You can also run the `train.py` script to train the models and extract results.

## Models Used

- **SGD Regression**
- **Ridge Regression**
- **K-Nearest Neighbors**
- **Random Forest**

## Results

The Random Forest model demonstrated the highest performance with an R2 Score of 0.748, indicating that it can explain approximately 74.8% of the variance in car prices.

## Future Work

- Incorporate more sophisticated regression models.
- Implement hyperparameter tuning using methods like Random Search and Bayesian Optimization.
- Explore ensemble methods to potentially improve prediction accuracy.

## Contributors

- Vasileios Vitzilaios (sent me an [e-mail](mailto:vasileios.vitzilaios@icloud.com))
