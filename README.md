# meta-net
Formerly meta-net, now classifying German street signs using convolutional
neural networks!

## Data source
The data used can be found at
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html.
For the training data, we utilized GTSRB_Final_Training_Images.zip and for
the testing data we utilized GTSRB_Final_Test_Images.zip. The labels for the
test data can be found in GTSRB_Final_Test_GT.zip.

## Work Log
12/8/2019: Gareth (4 Hours)
  1) Wrote code to load in the training and testing data using cv2, csv, and
  numpy
  2) Wrote code to normalize the image data and convert it to 32 x 32

12/8/2019: Gareth (2 Hours)
  1) Wrote some code to plot label distributions for training and test data
  2) Wrote code for basic convnet architecture, to be replaced later

12/9/2019: Gareth (2 Hours)
  1) Converted labels to numpy arrays, then created tensorflow datasets
  2) Added code for model training and testing (no validation split yet)

## Todo
1) Image transformations, shouldn't be too difficult with tensorflow datasets?
2) Model architecture
3) Regularization parameters
4) Model quantization 
