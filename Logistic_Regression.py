# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression

# Generate a dataset
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples = 80,
                  centers = 3,
                  random_state = 2,
                  cluster_std = 0.75)

# Show the dataset 
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'Accent')

def classificator_areas_plot(model, x, y, name_model, test_idx = None):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from matplotlib.colors import ListedColormap
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_dark = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                                
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                         np.arange(y_min, y_max, 0.05))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    
    plt.pcolormesh(xx, yy, z, cmap = cmap_light)
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap = cmap_dark)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('The model is ' + name_model)
    
    if test_idx != None:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1],
                    edgecolors = 'k',
                    facecolors = 'none',
                    label = 'Test')
    
# Load the model
logic_regre = LogisticRegression().fit(x, y)

classificator_areas_plot(logic_regre, x, y, 'Logistic Regression')

predict = logic_regre.predict(x)

from sklearn.metrics import accuracy_score

accu_log_regre = accuracy_score(y, predict)

print('La precision del algoritmo Regresión Logística es ', round(accu_log_regre, 2))