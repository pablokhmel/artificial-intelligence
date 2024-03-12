import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
X = np.c_[np.ones((len(x_train), 1)), x_train]

theta_best = X.T.dot(X)
theta_best = np.linalg.inv(theta_best)
theta_best = theta_best.dot(X.T)
theta_best = theta_best.dot(y_train)

print(f'Closed-form theta: {theta_best}')

# TODO: calculate error
X = np.c_[np.ones((len(x_test), 1)), x_test]
pred = X.dot(theta_best)
mse = np.mean((pred - y_test) ** 2)

print(f'Closed-form error: {mse}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
z_x_train = (x_train - np.mean(x_train)) / np.std(x_train)
z_y_train = (y_train - np.mean(y_train)) / np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
def dif_mse(x, y, theta):
    X = np.c_[np.ones((len(x), 1)), x]
    res = X.dot(theta) - y
    res = X.T.dot(res)
    res = 2 / len(x) * res
    return res
    
theta = np.random.rand(2)


lr = 0.01
max_num = 1000
for i in range(max_num):
    theta = theta - lr * dif_mse(z_x_train, z_y_train, theta)


print(f'Batch Gradient Descent theta: {theta}')

# TODO: calculate error
z_x_test = (x_test - x_train.mean()) / x_train.std()
X = np.c_[np.ones((len(z_x_test), 1)), z_x_test]

y_test_pred = X.dot(theta)
y_test_pred = y_test_pred * y_train.std() + y_train.mean()

stand_mse = (1 / len(z_x_test)) * np.sum((y_test_pred - y_test) ** 2)

print(f'Error after standardization: {stand_mse}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
