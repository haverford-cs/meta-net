""" A file to plot the data that we used for the presentation
Authors: Gareth Nicholas + Emile Givental
Date: December 11th, 2019
"""

import matplotlib.pyplot as plt
import math
if __name__ == "__main__":
  params =   [1, 25211, 85547, 101995, 158891, 368683, 889291, 1621707, 3882027, 20958699]
  log_params = [math.log(param) for param in params]
  results = [54, 79.75455474853516, 81.77355194091797, 90.36421203613281, 91.58353424072266, 88.33728790283203, 95.45526123046875, 96.83293914794922 , 97.23674011230469, 97.600952148437] 
  
  rotating_results = [54, 83.49169158935547, 84, 91, 91.2, 92, 95, 96.9, 97.3, 97.8]
  
  plt.plot(log_params, results, 'r*-', label = "normal")
  plt.plot(log_params, rotating_results, 'g*-', label = "rotated")
  plt.xlabel("parameters (log)")
  plt.ylabel("accuracy")
  plt.legend()
  plt.title("Accuracy vs Number of Parameters")
  plt.show()