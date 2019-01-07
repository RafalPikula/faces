# Face as Gender and Age Determinant  
The main goal of this project is to learn and test how well different convolutional neural network architectures can deal with the problem of gender detection and/or age prediction when fed solely the photographs of faces. In the entire project we use keras with tensorflow backend.

Note that the data used for this project can be downloaded from this site: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
by clicking "Download faces only (7GB)" and "Download images meta data".

## Short Description
The Jupyter notebooks are numbered for easier navigation. The photo data requires some preprocessing and cleaning which is done in the first notebook while the second notebook contains the steps needed to prepare the train-validation-test split. Apart from these, there are four notebooks with different gender classification models and two notebooks with age prediction models. Finally, the last notebook provides an explanation of some seemingly substandard results. Note that due to its size the data was divided into 20 parts (unless stated otherwise), and the model training was carried out consecutively on each part at a time.

The upshot is that gender classification achieves good results nearing in best case 89% accuracy while age prediction is a much more demanding task - the mean absolute error of the best model (out of the tested models) in this category is merely 7.38 years, however, in the last notebook we theorize that under certain circumstances it might be possible to drive this value down, even below 5 years.

## More Detailed Description
The preprocessing and preparatory steps are well commented in the first two notebooks and will not be expanded upon. Instead, we will concentrate on outlining the models used and summarizing the results. We will also throw some light on the causes of the high age prediction error.

### Gender Classification Models
**Model G1** is a very simple sequential model with only 3 convolutional layers coupled with some max pooling and batch normalization layers (no data augmentation used) while the classifying top of the model uses dropout (0.5). The accuracy achieved is **85.186%**.

**Model G2** is a nonsequential model consisting of four blocks where each block has two concatenated branches of two convolutional layers intertwined with batch normalization layers and after each concatenation there is a max pooling and batch normalization layer (no data augmentation used) while the classifying top of the model uses dropout (0.5) and l2 regularization. The accuracy achieved is **85.474%**.

**Model G3** is a more complex version of Model G1 with 6 instead of 3 convolutional layers, the classifying top of the model uses dropout (0.5) and this time data augmentation was used to train the model. The accuracy achieved is **87.587%**.

**Model G4** is basically the VGG19 model without the top and with the weights trained on Imagenet, the custom classifying top uses dropout (0.25). The model was trained with data augmentation as follows: first the classifying top was trained on the first part of the data with the VGG19 layers frozen, then the high-level convolutional layers were unfrozen, and the model was trained on all parts of the data. The accuracy achieved is **88.938%**.

**Model G5** is a simpler version of Model G3 with a different and more efficient training approach employed leading to an improvement upon its 'bigger relative' - the accuracy of Model G5 is **88.532%**. In this particular case the data was divided into 100 parts.

### Age Prediction Models
**Model A1** is a more complex version of Model G3 above where each convolutional layer is replaced with a block of 2 such layers intertwined with batch normalization layers while the top of the model uses dropout (0.5). The model was trained with data augmentation. The results: the mean squared error is 95.946, the mean absolute error is **7.408**.

**Model A2** is again the VGG19 model without the top and with the weights trained on Imagenet, but with a different custom top without dropout. The model was trained with data augmentation. The results: the mean squared error is 95.629, the mean absolute error is **7.382**.

**Model A3** is a simpler version of Model A1 trained using a different training approach. the results: the mean squared error is 99.406, the mean absolute error is **7.423**. In this particular case the data was divided into 100 parts.

### High Mean Absolute Error Illusion
The error of 7.382 years is quite big, and it may seem that the model didn't do a good job. However, in the last notebook we compute and visualize the mean absolute error per age and the number of photographs in the train set per age. It turns out that as the number of photos decreases the error increases and that these two variables are very strongly negatively correlated (-0.8825). In other words, the model doesn't learn well for those age values that are highly underrepresented in the train set.

If, on the other hand, we restrict our attention to the well represented \[26, 45\] age bracket, the results are much better - the mean absolute error drops to **4.921**. It can be concluded that the unbalanced nature of the age distribution in the train set degrades the model performance. Therefore, it is conceivable that artificially balancing the age distribution (and effectively making it uniform) would have a positive impact on the model performance.

For completeness, we include the plots of predictions vs. actual age values and their joint density for both the full age range of the train set (the \[11, 70\] age bracket) and the \[26, 45\] age bracket.

We start with the full age range:

![Predicted Age vs. Actual Age for Full Age Range](https://github.com/RafalPikula/faces/blob/master/pictures/Predicted_Age_vs_Actual_Age_for_Full_Age_Range.png)

![Joint Density of Predicted and Actual Age with Error Bands for Full Age Range](https://github.com/RafalPikula/faces/blob/master/pictures/Joint_Density_of_Predicted_and_Actual_Age_with_Error_Bands_for_Full_Age_Range.png)

Then, we restrict our attention to the \[26, 45\] age bracket:

![Predicted Age vs. Actual Age for Age Bracket](https://github.com/RafalPikula/faces/blob/master/pictures/Predicted_Age_vs_Actual_Age_for_Age_Bracket.png)

![Joint Density of Predicted and Actual Age with Error Bands for Age Bracket](https://github.com/RafalPikula/faces/blob/master/pictures/Joint_Density_of_Predicted_and_Actual_Age_with_Error_Bands_for_Age_Bracket.png)

