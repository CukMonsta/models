# -*- coding: utf-8 -*-

### Linear Regression with one feature

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset 

from sklearn.datasets import load_boston

dataset= load_boston(return_X_y=True)

# Split values to train and test
from sklearn.model_selection import train_test_split

X = pd.DataFrame(dataset[0][:,0])
Y = pd.DataFrame(dataset[1])

seed = 2
sizetest = 0.2

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = sizetest, random_state = seed) 

# Create a linear regression model

model = LinearRegression(fit_intercept=False)
model.fit(x_train,y_train)

# Predict with our model

predict = model.predict(x_test)

# Parametros de ajuste

#beta_0 = model.intercept_[0] #termino independiente debe ser >0
beta_1 = model.coef_[0][0]

r_2_train = model.score(x_train, y_train)
r_2_test = model.score(x_test, y_test)

# Accuracy's model

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predict)

# plot the model

import matplotlib.pyplot as plt

list_plot1 = [x_test.iloc[0,0], x_test.iloc[60,0]]
list_plot2 = [min(predict), max(predict)]

plt.figure('Linear Model')
plt.scatter(x_test.iloc[0:25,0], y_test.iloc[0:25,:], label = 'data')
plt.plot(list_plot1, list_plot2, label = 'model', c = 'red')
plt.legend(loc = 1)
plt.title('Linear Regression Model')
plt.show()

#print('El valor de beta_0 es : ', beta_0)
print('El valor de beta_1 es: ', round(beta_1, 2))
print('El valor del coefciente R^2 en el training es: ', round(r_2_train, 2))
print('El valor del coefciente R^2 en el test es: ', round(r_2_test, 2))
print('Tras realizar predicciones con el modelo el MSE ha sido de: ', round(mse, 2))
print('Este problema no se puede resolver eficientemente mediante un algoritmo lineal')
print('El error de la predicción es ', round(mse, 2))