""" A argument parsing file + confusion matrix creation 
Authors: Gareth Nicholas + Emile Givental
Date: December 11th, 2019
"""

import optparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 

def parse_args():
  parser = optparse.OptionParser(description='run final project')
  model_list = ["convnet", "reduced_convnet", "multi_scale_conv", "KNN",\
   "reduced_dense_256", "reduced_dense_256_pool", "reduced_dense_and_conv", \
   "reduced_dense_and_conv2", "reduced_dense_and_conv3", "tiny_conv", "miniscule_conv"]
  parser.add_option('-m',
  '--model', type='string', default = "tiny_convnet", help = "the options for model are" + str(model_list))
     
  parser.add_option('-r', action="store_true", dest="rotating", default = False, help= "whether to rotate the data set (always both directions 12 degrees), default is [default]")
  parser.add_option('-s', action="store_true", dest="shifting", default = False, help = "whether or not to shift the data, default is [default]")
  parser.add_option('-v', action="store_true", dest="verbose", default = False, help = "whether or not to print results to cmdline, default = [default]")
  parser.add_option('-l', action="store_true", dest="validation", default = False, help = "whether or not to use a validation dataset for hyperparameter (NOT YET INSTALLED), default = [default]")
  parser.add_option('-a', action="store_true", dest="saving", default = False, help= "whether or not to save the model, default = [default]")
  (opts, args) = parser.parse_args()
  
  if opts.model not in model_list:
    print(f"The model you chose does not exist, and must be in the list: \n" + str(model_list))
    exit()

  return opts

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
  x = df_confusion.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df = pandas.DataFrame(x_scaled, columns=df_confusion.columns)
  
  plt.matshow(df_confusion, cmap=cmap) # imshow
  plt.colorbar()
  tick_marks = np.arange(len(df_confusion.columns))
  plt.xticks(tick_marks, df_confusion.columns, rotation=45)
  plt.yticks(tick_marks, df_confusion.index)
  #plt.tight_layout()
  plt.ylabel(df_confusion.index.name)
  plt.xlabel(df_confusion.columns.name)
  
  plt.show()
