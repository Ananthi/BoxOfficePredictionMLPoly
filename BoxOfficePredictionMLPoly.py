from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE_ERROR = 0.8
X, y = load_data()
X_features = [
    "is_Action",
    "is_Adventure",
    "is_Fantasy",
    "is_Science Fiction",
    "is_Crime",
    "is_Drama",
    "is_Thriller",
    "is_Animation",
    "is_Family",
    "is_Western",
    "is_Comedy",
    "is_Romance",
    "is_Horror",
    "is_Mystery",
    "is_History",
    "is_War",
    "is_Music",
    "is_Documentary",
    "is_Foreign",
]

# Split Data
print(f"Overall data set size: {X.shape}")
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
X_test, X_cv, y_test, y_cv = train_test_split(X_, y_, test_size=0.20, random_state=1)
print(f"Training data set size: {X_train.shape}")
print(f"Validation data set size: {X_cv.shape}")
print(f"Test data set size: {X_test.shape}")

# Initialize the class
linear_model = LinearRegression()
# Train the model
linear_model.fit(X_train, y_train)

# Feed the training set and get the predictions
yhat = linear_model.predict(X_train)
# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# Feed the validation set and get the predictions
yhat_cv = linear_model.predict(X_cv)
print(
    f"Validation MSE (using sklearn function): {mean_squared_error(y_cv, yhat_cv) / 2}"
)

# Feed the test set and get the predictions
yhat_test = linear_model.predict(X_test)
print(f"Test MSE (using sklearn function): {mean_squared_error(y_test, yhat_test) / 2}")


# Find best polynomial
# Instantiate the class to make polynomial features
train_mses = []
cv_mses = []
for d in range(1, 8):
    print(f"POLY DEGREE: {d}")
    poly = PolynomialFeatures(degree=d, include_bias=False)

    # Compute the number of features and transform the training set
    X_train_mapped = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_mapped, y_train)
    yhat = model.predict(X_train_mapped)
    # Use scikit-learn's utility function and divide by 2
    train_mse = mean_squared_error(y_train, yhat) / 2

    print(f"training MSE (using sklearn function): {train_mse}")
    train_mses.append(train_mse)

    # Feed the validation set and get the predictions
    X_cv_mapped = poly.fit_transform(X_cv)
    yhat_cv = model.predict(X_cv_mapped)

    cv_mse = mean_squared_error(y_cv, yhat_cv) / 2
    print(f"Validation MSE (using sklearn function): {cv_mse}")
    cv_mses.append(cv_mse)

    # # Feed the test set and get the predictions
    # yhat_test = model.predict(X_test)
    # print(
    #     f"Test MSE (using sklearn function): {mean_squared_error(y_test, yhat_test) / 2}"
    # )
degrees = range(1, 11)
plt.plot(degrees, train_mses, color="r")
plt.plot(degrees, cv_mses, color="g")
plt.title("degree of polynomial vs. train and CV MSEs")
plt.show()
