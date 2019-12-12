# meta-net
Formerly meta-net, now classifying German street signs using convolutional
neural networks!

Run with python3 driver.py

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

12/9/2019: Gareth (2 Hours)
  1) Added multi-scale model with regularization, might want batch norm
  2) Bumped up number of epochs, might want to adjust stopping criteria
  3) Removed validation set for now
  4) Added model statistics and architectures
  Notes: Test accuracy for the new model is around 95%, but number of params
  is much lower (around 5 million). I removed the validation set since the
  model had poor accuracy on it (much lower than the test acc). It might be
  more useful to use cross-validation instead.

12/9/2019: Gareth (4 Hours)
  1) Added reduced convnet model which gets us down to 1.6 million parameters
  and almost 97% accuracy!!!
  2) Recalculated some of the model benchmarks with different amounts of
  regularization, added batch normalization to multi-scale.
  3) Added additional models for graph of params vs accuracy

12/10/2019: Gareth (1 Hour)
  1) Added extra models for comparison. Their statistics can be found below
  and the models are in comparison_models.py

12/11/2019: Gareth (1 Hour)
  1) Added code to plot confusion matrix as a heatmap

12/12/2019 Gareth (1 Hour)
  1) Code clean up and commenting

## Working with GPUs and lab computers
It is currently impossible to commit from the lab computers to remote without
generating a new SSH key (if you have TFA enabled). Thus I would suggest
scp -r to move all the files between commits. You only want to move the data
once, so scp -r ~/path_to_files/*.py will move just the code files.

Also if you need to get around the lab computers not having opencv, I used
pip install --user opencv-python.

## Current Model Statistics (12/9/2019)
Models were trained for 20 epochs each.

### Class competition model: convnet()

Total params: 20,960,619
Trainable params: 20,958,699
Non-trainable params: 1,920

Test Loss: 0.2239145189523697, Test Accuracy: 97.22090148925781

### Multi-scale cnn: multi_scale_conv()

Total params: 3,882,475
Trainable params: 3,882,027
Non-trainable params: 448

Test Loss: 0.7362900376319885, Test Accuracy: 96.46080780029297

### Smaller version of class competition model (SCCM): reduced_convnet()

Total params: 1,622,603
Trainable params: 1,621,707
Non-trainable params: 896

Test Loss: 0.23565997183322906, Test Accuracy: 96.95170593261719

### SCCM but with less nodes in the dense layers (256 vs 512): reduced_dense_256()

Total params: 890,187
Trainable params: 889,291
Non-trainable params: 896

Test Loss: 0.27640843391418457, Test Accuracy: 96.76959228515625

### Model above with larger filter sizes, reduced the number of convolutional layers: reduced_dense_256_pool()

Total params: 368,875
Trainable params: 368,683
Non-trainable params: 192

Test Loss: 0.40244945883750916, Test Accuracy: 93.11164093017578

### Model above with half the number of filters: reduced_dense_and_conv()

Total params: 158,987
Trainable params: 158,891
Non-trainable params: 96

Test Loss: 0.41992634534835815, Test Accuracy: 92.57323455810547

### Model above with half the number of filters again: reduced_dense_and_conv2()

Total params: 102,043
Trainable params: 101,995
Non-trainable params: 48

Test Loss: 0.6126990914344788, Test Accuracy: 90.41963958740234

### Halve the filters once again (down to 1/8 of original): reduced_dense_and_conv3()

Total params: 85,571
Trainable params: 85,547
Non-trainable params: 24

Test Loss: 0.6464730501174927, Test Accuracy: 88.0680923461914

### Tiny model to get corner of our graph: tiny_conv()

Total params: 25,227
Trainable params: 25,211
Non-trainable params: 16

Test Loss: 0.8594541549682617, Test Accuracy: 80.68091583251953

### Going even smaller than the last model: miniscule_conv()

Total params: 2,284
Trainable params: 2,278
Non-trainable params: 6

Test Loss: 1.25192391872406, Test Accuracy: 69.57244873046875

Wow conv nets are incredible.

## References

Generating confusion matrix heatmap: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
Normalizing confusion matrix: https://stackoverflow.com/questions/35678874/normalize-rows-of-pandas-data-frame-by-their-sums/35679163
Data preprocessing on this dataset: https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter6/datasets/gtsrb.py
