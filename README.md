# Face as Gender and Age Determinant  
The main goal of this project is to learn and test how well different convolutional neural network architectures can deal with the problem of gender detection and/or age prediction when fed solely the photographs of faces. In the entire project we use keras with the tensorflow backend.

Note that the data used for this project can be downloaded from this site: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
by clicking "Download faces only (7GB)" and "Download images meta data".

## Short Description
The Jupyter notebooks are numbered for easier navigation. The photo data requires some preprocessing and cleaning which is done in the first notebook while the second notebook contains the steps needed to prepare the train-validation-test split. Apart from these, there are five notebooks dealing with gender classification: four notebooks with different models plus one with model ensemble and three notebooks dealing with age prediction: two notebooks with age prediction models plus one with model ensemble. Finally, the last notebook provides some thoughts on age prediction results.

The upshot is that gender classification achieves good results nearing in best case **90%** accuracy while age prediction is a much more demanding task - the best mean absolute error achieved is approximately **6.35** years. However, in the last notebook we shed some light on a specific factors that influences this score and theorize that under certain circumstances it might be possible to drive this value down, even below **4.5** years.

## More Detailed Description
The preprocessing and preparatory steps are well commented in the first two notebooks and will not be expanded upon. Instead, we will concentrate on outlining the models used and summarizing the results. Note that data augmentation is used throught model training and that in case of gender classification the class weights are used in order to mitigate the (even if small) class imbalance influence.

### Gender Classification Models
**Model G1** is a relatively simple custom sequential model with only 6 convolutional layers coupled with max pooling and batch normalization layers while the classifying top of the model uses dropout (0.5). The accuracy achieved is **88.925%**.

**Model G2** is the ResNet50 model available in Keras while the classifying top of the model uses dropout (0.5). This model is first trained for several epochs with ResNet50 layers set to be non-trainable and subsequently fine-tuned by setting a certain numbers of previously non-trainable layers to be trainable. The accuracy achieved is **88.256%**.

**Model G3** is the Xception model available in Keras while the classifying top of the model uses dropout (0.5). This model is first trained for several epochs with ResNet50 layers set to be non-trainable and subsequently fine-tuned by setting a certain numbers of previously non-trainable layers to be trainable. The accuracy achieved is **88.794%**.

**Model G4** is a more complicated version of the Model G1 encompasing 6 convolutional blocks of layers each containing 2 convolutional layers intertwined with max pooling, batch normalization and spatial dropout layers while the classifying top of the model uses dropout (0.5). The accuracy achieved is **89.476%**.

**Model Ensemble** is a simple ensemble of the above 4 models obtained via the arithmetic average of class probabilities predicted by the constituent networks. The accuracy achieved is **89.949%**.

### Age Prediction Models
**Model A1** uses the same architecture as the Model G4 adjusted for regression modelling. The results: the mean squared error is 80.393, the mean absolute error is **6.444**.

**Model A2** is a version of Model A1 with some minor differences (rescaling, different activation function in the last dense layer). The results: the mean squared error is 80.212, the mean absolute error is **6.476**.

**Model Ensemble** is a simple ensemble of the above 2 models obtained via the arithmetic average of the predicted values. The results: the mean squared error is 78.197, the mean absolute error is **6.351**.

### Some thoughts on Age Prediction Model Performance
The age prediction error of 6.351 years is not bad but it could be better. In the last notebook we compute and visualize the mean absolute error per age and the number of photographs in the train set per age. It turns out that as the number of photos decreases the error increases and that these two variables are very strongly negatively correlated (-0.8653). In other words, it seems that these models do not learn well for those age values that are highly underrepresented in the train set.

If, on the other hand, we restrict our attention to the well represented \[26, 45\] age bracket, the results are much better - the mean absolute error drops to **4.463**. It can be concluded that the unbalanced nature of the age distribution in the train set potentially degrades the model performance. Therefore, it is conceivable that artificially balancing the age distribution (and effectively making it uniform) would have a positive impact on the model performance.

For completeness, we include the plots of predictions vs. actual age values and their joint density for both the full age range of the train set (the \[11, 70\] age bracket) and the \[26, 45\] age bracket.

We start with the full age range:

![Predicted Age vs. Actual Age for Full Age Range](https://github.com/RafalPikula/faces/blob/master/pictures/Predicted_Age_vs_Actual_Age_for_Full_Age_Range.png)

![Joint Density of Predicted and Actual Age with Error Bands for Full Age Range](https://github.com/RafalPikula/faces/blob/master/pictures/Joint_Density_of_Predicted_and_Actual_Age_with_Error_Bands_for_Full_Age_Range.png)

Then, we restrict our attention to the \[26, 45\] age bracket:

![Predicted Age vs. Actual Age for Age Bracket](https://github.com/RafalPikula/faces/blob/master/pictures/Predicted_Age_vs_Actual_Age_for_Age_Bracket.png)

![Joint Density of Predicted and Actual Age with Error Bands for Age Bracket](https://github.com/RafalPikula/faces/blob/master/pictures/Joint_Density_of_Predicted_and_Actual_Age_with_Error_Bands_for_Age_Bracket.png)

