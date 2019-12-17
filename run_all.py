import subprocess
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
  model_list = ["tiny_conv", "reduced_dense_and_conv", "reduced_dense_and_conv2", "reduced_dense_and_conv3",
     "reduced_dense_256_pool", "reduced_dense_256",  "reduced_convnet", "multi_scale_conv", "convnet"]
     
  #for model in model_list:
    #subprocess.call(["python3", "driver.py", "-m", model, "-a"])
  
  for model in model_list: 
    subprocess.call(["python3", "driver.py", "-m", model, "-a", "-r"])
  
  