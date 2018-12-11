# Face as Gender and Age Determinant  
The main goal of this project is to learn and test how well different convolutional neural network architectures can deal with the problem of gender detection and/or age prediction when fed solely the photographs of faces. 

Note that the data used for this project can be downloaded from this site: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
by clicking "Download faces only (7GB)" and "Download images meta data".

### Short Description
The Jupyter notebooks are numbered for easier navigation. The photo data requires some preprocessing and cleaning which is done in the first notebook while the second notebook contains the steps needed to prepare the train-validation-test split. Apart from these there are four notebooks with different gender classification models and two (so far) notebooks with age prediction models.

The upshot is that gender classification achieves good results nearing in best case 89% accuracy while age prediction is a much more demanding task - the mean absolute error of the best model (out of the tested models) in this category is merely 7.38 years.

### More Detailed Description
The preprocessing and preparatory steps are well commented in the first two notebooks and will not be expanded upon. Instead, we will concentrate on outlining the models used and summarizing the results.

######Gender Classification
Model 1
