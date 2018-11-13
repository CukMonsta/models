# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generate a dataset
x, y = make_circles(n_samples = 100,
                  factor = 0.1,
                  noise = 0.1,
                  random_state= 1)

plt.scatter(x[:, 0], x[:, 1], c= y, cmap = 'Accent')

model = SVC().fit(x,y)

# with the next function we'll define the edge of the classifier
def SVC_separator(x, y, model, name_model):
    def plot_svm_separator(model, x, y, name_model):
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
        xx = np.linspace(x_min, x_max, 10)
        yy = np.linspace(y_min, y_max, 10)
    
        X1, X2 = np.meshgrid(xx, yy)
        z = np.empty(X1.shape)
    
        for (i, j), val in np.ndenumerate(X1):
            x1 = val
            x2 = X2[i, j]
            p = model.decision_function(np.array([[x1, x2]]))
            z[i, j] = p[0]
        
        levels = [-1.0, 0, 1.0]
        linestyles = ['dashed', 'solid', 'dashed']
        colors = 'k'
    
        plt.contour(X1, X2, z, levels, colors = colors, linestyles = linestyles)
        plt.scatter(x[:, 0], x[:, 1], c= y, cmap= 'Accent')
        plt.show()
    
    from sklearn.svm import SVC

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for i in range(len(kernels)):
        model = SVC(kernel= kernels[i]).fit(x, y)
        plt.subplot(2, 2, 1 + i)
        plot_svm_separator(model, x, y, 'SVM')
        plt.title(kernels[i])
        plt.show()
                    
    print('The model is ', name_model)


SVC_separator(x, y, model, 'SVC')

predict = model.predict(x)

from sklearn.metrics import accuracy_score

accu_SVC = accuracy_score(y, predict)
print('La precision del modelo es ', round(accu_SVC, 2))