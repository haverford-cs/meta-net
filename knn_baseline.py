from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from util import plot_confusion_matrix


class KNN:
  def __init__(self, X, y, n=4, max_data = 10000):
    self.model = KNeighborsClassifier(n_neighbors = n)
    self.model.fit(X[0:100000],y[0:100000])
    
  def predict(self, X, y):
    predictions = self.model.predict(X)
    acc = sum([1 if predictions[i] == y[i] else 0 for i in range(len(predictions))])/len(predictions)
    print(f"The accuracy was {acc}")
    y_actu = pd.Series([i for i in y], name = 'Actual')
    y_pred = pd.Series([i for i in predictions], name = 'Predicted')
    df_confusion = pd.crosstab(y_pred, y_actu)
    df_norm = df_confusion / df_confusion.sum(axis=1)
    plot_confusion_matrix(df_norm)
    return predictions
    
