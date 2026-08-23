# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
• Import the required Python libraries.

• Load the weather station dataset.

• Convert the time column into datetime format.

• Extract month, day, and hour features from the time column.

• Handle missing values in the dataset.

• Select humidity, pressure, wind speed, month, day, and hour as input features.

• Select temperature, PM2.5, and energy values as target variables.

• Split the dataset into training and testing datasets.

• Create the Random Forest Regression model.

• Train the model using training data.

• Predict the output using testing data.

• Evaluate the model using R² Score, MAE, and RMSE.

• Perform 5-fold cross-validation to verify model stability.

• Display the prediction accuracy and final results. 

## Program:
```
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

# Display first five rows
print("First Five Rows of Dataset")
print(df.head())

# Convert time column into datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date and time features
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Handle missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Select input features
X = df[['hum', 'pressure', 'wind_speed', 'month', 'day', 'hour']]

# Select target variables
Y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, Y_train)

# Predict output values
Y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Display evaluation metrics
print("\nModel Evaluation Metrics")
print("R2 Score :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# Perform cross-validation
cv_scores = cross_val_score(
    model,
    X,
    Y,
    cv=5,
    scoring='r2'
)

# Display cross-validation results
print("\nCross Validation Scores")
print(cv_scores)

print("\nAverage Cross Validation Score")
print(cv_scores.mean())
Developed by:  YUVASRI K P
RegisterNumber:  212225060314

```

## Output:
<img width="1727" height="689" alt="image" src="https://github.com/user-attachments/assets/9279ed91-3295-4a62-a053-59a5ef2a0c40" />


## Result:
Thus the program to implement the Random Forest Algorithm for Predicting the Weather is written and verified using python programming.
