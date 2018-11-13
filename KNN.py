# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

dataset = make_classification(n_samples = 1000, 
                              n_features = 8,
                              n_informative = 6,
                              n_redundant = 2, 
                              n_repeated = 0,
                              n_classes = 3,
                              n_clusters_per_class = 2,
                              weights = None, 
                              class_sep = 1, 
                              random_state = 5)

x = dataset[0]
y = dataset[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

radio = 3

model_radius = RadiusNeighborsClassifier(radius = radio)

model_radius.fit(x_train, y_train)

predict_radius = model_radius.predict(x_train)

accu_radius = accuracy_score(y_train, predict_radius)

print('La precision del modelo KNN Radius es ', round(accu_radius, 2))

from sklearn.neighbors import KNeighborsClassifier

neighbors = 4

model_KNN = KNeighborsClassifier(n_neighbors = neighbors, 
                             n_jobs = 2)

model_KNN.fit(x_train, y_train)

predict_KNN = model_KNN.predict(x_test)

accu_KNN = accuracy_score(y_test, predict_KNN)

print('La precision del modelo KNN es ', round(accu_KNN, 2))

print('Generalmente la precision obtenida con el predict utilizando el train ',
      'será mayor que con el test')
