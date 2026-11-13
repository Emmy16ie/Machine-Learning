import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing(as_frame = True)

X = data.data
y = data.target

print("Rows: ", X.shape)
print("\n Features")
print(X.head())

#Split data

X_train, X_test, y_train, y_test = train_test_split( X , y , test_size=0.2, random_state=42)

print("Train size: ", X_train.shape)
print("Test size: ", X_test.shape)

# Creating the model
model = LinearRegression()
model.fit(X_train,y_train)

# Make prediction
y_pred = model.predict(X_test)

#Evaluate Model
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n --- Model Evaluation ---")
print("\n MAE = ", mae)
print("\n MSE = ", mse)
print("\n RMSE = ", rmse)
print("\n R2 = ", r2)

#Print The Model Parameters
print("\n --- Model Parameter ---")
print("\n ------------------------")

for features,weights in zip(X.columns, model.coef_):
    print(f"{features} : {weights}")

print("\n Bias: ", model.intercept_)

#Compare prediction to actual Values
results = pd.DataFrame(
    {
        "Actual: " : y_test.values,
        "Predicted: " : y_pred
    }
) 

print("\n --- Predictions ---")
print(results.head(10))

#Plot Actual vs Predicted
plt.figure(figsize = (8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted values vs. Actual values")

plt.show()

#Save trained Model
joblib.dump(model, "models/house_price_model.pkl")

print("\n Prediction Model Saved.")