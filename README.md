# Face as Gender and Age Detrminant  
The main goal of this project is to test how well different convolutional neural network achitectures can deal with the problem of gender detection and/or age prediction when fed solely the photographs of faces. 

Note that the data used in this project can be downloaded from this site: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
by clicking "Download faces only (7GB)" and "Download images meta data".

### Short Description
The photo data requires some preprocessing and cleaning which is done in the first notebook while the second notebook contains the steps needed to prepare the train-validation-test split. Apart from these there are four noteboooks with different gender classification models and two notebooks with age predicton models.

The upshot is that gender classification achieves good results nearing 90% accuracy while age prediction is a much more demanding task - the mean absolute error is slightly below 7.4 years.
