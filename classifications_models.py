# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

# Definiremos una función para analizar las metricas de los modelos de clasificacion
def metrics_model_clasi(y_true, y_pred, name_model, n_plots):
    #You can load the library from here
#    from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
    
    print('The model is', name_model)
    
    # confusion matrix
    confu_matrix = confusion_matrix(y_true, y_pred)
    
    print('Matriz de confusion: ', confu_matrix)
    
    #Indicadores
    accura = accuracy_score(y_true, y_pred) #precision
    print('The accuracy of the model is: ', round(accura, 2))
    
    precision = precision_score(y_true, y_pred) #exactitud
    print('The precision of the model is: ', round(precision, 2))
    
    recall = recall_score(y_true, y_pred) #exhaustividad
    print('The recall of the model is: ', round(recall, 2))
    
    f1 = f1_score(y_true, y_pred)
    print('F1', round(f1, 2))
    
    #Roc Curve
    false_positive_r, rcl, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_r, rcl)
    
    print('AUC', round(roc_auc, 2))

    plt.figure(n_plots)
    plt.plot(false_positive_r, rcl, c = 'b', label = 'model')
    plt.plot([0, 1], [0, 1], 'r--', label = 'frontier')
    plt.title('AUC = %0.2f' % roc_auc + ' ' + name_model)
    plt.legend()
    plt.show()
    
#    return confu_matrix, accura, precision, recall, f1, roc_auc

    
# Load or generate a dataset
# Problem with two random features
y_true = np.round(np.random.uniform(size = 20), 0)
y_pred = np.round(np.random.uniform(size = 20), 0)

metrics_model_clasi(y_true, y_pred, 'Problem with two features', 1)

print('######################################################################')

# Problem with X features
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset
x, y = make_classification(n_samples= 2000,
                           n_features= 10,
                           n_redundant= 0)

#Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 5)

# Compare a few models
# Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression().fit(x_train, y_train)

predict = classifier.predict(x_test)

#metrics_model_clasi(y_train, classifier.predict(x_train), 'Logistic Regression') # Para comprobar el resultado del train y el test
metrics_model_clasi(y_test, predict, 'Logistic Regression', 2)

print('######################################################################')

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier().fit(x_train, y_train) # we can try to avoid the overfitting with DecisionTreeClassifier(max_depth = )

predict_tree = tree.predict(x_test)

#metrics_model_clasi(y_train, tree.predict(x_train), 'Decision Tree') # Para comprobar el resultado del train y el test
metrics_model_clasi(y_test, predict_tree, 'Decision Tree', 3)

# Feature importances from the tree
print('El peso de las variables en el modelo decision tree es el siguiente: ', np.round(tree.feature_importances_, 3))

print('######################################################################')

# Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier().fit(x_train, y_train) # we can add RandomForestClassifier(max_depth = x)
pred_forest = forest.predict(x_test)

metrics_model_clasi(y_test, pred_forest, 'Random Forest', 4)

print('El peso de las variables en el modelo de forest es el siguiente: ', np.round(forest.feature_importances_, 3))

print('######################################################################')

# SVM
from sklearn.svm import SVC

svc_classi = SVC().fit(x_train, y_train)
pred_svc = svc_classi.predict(x_test)

metrics_model_clasi(y_test, pred_svc, 'SVM', 5)

print('######################################################################')

# Naïve Bayes
from sklearn import naive_bayes

naive_classi = naive_bayes.GaussianNB().fit(x_train, y_train)
pred_naive = naive_classi.predict(x_test)

metrics_model_clasi(y_test, pred_naive, 'Naïve Bayes', 6)

