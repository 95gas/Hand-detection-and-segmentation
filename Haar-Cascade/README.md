# Hand-Detection-and-Segmentation
 
FIRST APPROACH 
We tried to train a Cascade classifier with a window size of 30x60. 
We couldn't achieve good results by using this approach since we get a lot of false positive predicted. 
This was almost expected considering that the Haar Cascade classifier cannot manage to detect object of different viewpoint. Better results using this approach can be obtained by training three Haar classifier in order to try to predict hands in different viewpoint: 
 - a classifier for prediciting hands in vertical orientation. Hence we use a vertical rectangle window size as the one we used such as 30x60
 - a classifier for predicting hands in orrizontal orientation. Hence we need an orizontal rectangle window size such as 60x30
 - a classifier for predicting hands in oblique orientation. We might try using a square window size such as 24x24. 

However, given the lack support for gpu computation of the built-in opencv function for training the classifier, training 3 classifier will require some days. 
