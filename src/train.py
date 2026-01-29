import pandas
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from visualize import visualize

df = pandas.read_csv("../data/mpg_data.csv")
X = df[["Horsepower"]].values
y = df["MPG"].values
slope, intercept, r, p, std_err = stats.linregress(X, y)


def my_func(x):
    return slope * x + intercept


X_test, X_train, y_test, y_train = train_test_split(
    X, y, train_size=0.8, random_state=42
)
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# Matrics
# R2 score
r2 = r2_score(y_test, y_predict)
# Mean absolute error
mae = mean_absolute_error(y_test, y_predict)
# Mean squared error
mse = mean_squared_error(y_test, y_predict)
# Printing
print("Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
visualize(X, y, model)
