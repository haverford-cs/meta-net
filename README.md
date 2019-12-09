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
  2) Added code for model training and testing
  3) Added validation split
  4) Adjusted model architecture to work with new classes (43 instead of 10)
  Notes: The validation split seems a bit useless so we might want to remove
  it. The current model has a testing accuracy of 96%, not bad at all! However,
  the model is huge.

## Todo
1) Image transformations, shouldn't be too difficult with tensorflow datasets?
2) Model architecture
3) Regularization parameters
4) Model quantization
5) Save model

## Working with GPUs and lab computers
It is currently impossible to commit from the lab computers to remote. Thus
I would suggest scp -r to move all the files between commits. You only want
to move the data once, so scp -r ~/path_to_files/*.py will move just the code
files.

Also if you need to get around the lab computers not having opencv, I used
pip install --user opencv-python. 
