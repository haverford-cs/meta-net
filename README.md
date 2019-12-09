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

12/9/2019: Gareth (2 Hours)
  1) Added multi-scale model with regularization, might want batch norm
  2) Bumped up number of epochs, might want to adjust stopping criteria
  3) Removed validation set for now
  4) Added model statistics and architectures
  Notes: Test accuracy for the new model is around 95%, but number of params
  is much lower (around 5 million). I removed the validation set since the
  model had poor accuracy on it (much lower than the test acc). It might be
  more useful to use cross-validation instead.

## Todo
1) Image transformations, shouldn't be too difficult with tensorflow datasets?
2) Model quantization
3) Save model
4) Stopping criteria?
5) Batch norm?

## Working with GPUs and lab computers
It is currently impossible to commit from the lab computers to remote. Thus
I would suggest scp -r to move all the files between commits. You only want
to move the data once, so scp -r ~/path_to_files/*.py will move just the code
files.

Also if you need to get around the lab computers not having opencv, I used
pip install --user opencv-python.

## Current Model Statistics (12/9/2019)
Models were trained for 20 epochs each.
### Low param model:
__________________________________________________________________________________________________
Layer (type)                    Output Shape          Params     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 32)   2432        input_1[0][0]                    
__________________________________________________________________________________________________
activation (Activation)         (None, 32, 32, 32)   0           conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 16, 16, 32)   0           activation[0][0]                 
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 16, 16, 64)   32832       max_pooling2d[0][0]              
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 16, 16, 64)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 8, 8, 64)     0           activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 8, 8, 128)    131200      max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 8, 8, 128)    0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 4, 4, 32)     0           max_pooling2d[0][0]              
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 4, 4, 64)     0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 4, 4, 128)    0           activation_2[0][0]               
__________________________________________________________________________________________________
flatten (Flatten)               (None, 512)          0           max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1024)         0           max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 2048)         0           max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 3584)         0           flatten[0][0]                    
                                                                 flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1024)         3671040     concatenate[0][0]                
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         1049600     dense[0][0]                      
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 43)           44075       dropout_1[0][0]                  
==================================================================================================
Total params: 4,931,179
Trainable params: 4,931,179
Non-trainable params: 0

Test Loss: 0.558156430721283, Test Accuracy: 96.0094985961914

### Class competition Model
_________________________________________________________________
Layer (type)                 Output Shape              Params   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
activation (Activation)      (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
activation_3 (Activation)    (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
activation_4 (Activation)    (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584    
_________________________________________________________________
activation_5 (Activation)    (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 4, 4, 256)         131328    
_________________________________________________________________
activation_6 (Activation)    (None, 4, 4, 256)         0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 256)         262400    
_________________________________________________________________
activation_7 (Activation)    (None, 4, 4, 256)         0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 256)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 2, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 4000)              4100000   
_________________________________________________________________
dense_1 (Dense)              (None, 4000)              16004000  
_________________________________________________________________
dense_2 (Dense)              (None, 43)                172043    
=================================================================
Total params: 20,960,619
Trainable params: 20,958,699
Non-trainable params: 1,920

Test Loss: 0.1295158863067627, Test Accuracy: 97.5534439086914
