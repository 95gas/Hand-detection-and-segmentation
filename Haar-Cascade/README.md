# Hand-Detection-and-Segmentation
 
FIRST APPROACH 
We tried to train a Cascade classifier with a window size of 30x60 for detecting hands. 
We couldn't achieve good results by using this approach since we get a lot of false positive predicted. This bad result comes from the fact that the Haar Cascade classifier cannot manage to detect object of different viewpoint. Indeed, in our train dataset we consider hands of different orientation, but using a 30x60 window size for training on also orizontal oriented hands, includes deformation of the hand positive samples but also a lot of background gets included in the training positive samples.
For improving our solution we might need to go through the dataset and select only vertical oriented hands for training. In this case our classifier would be able to correctly detect only vertical hands.
We should hence train differen classifier for considering hands in different viewpoint: 
 - a classifier for prediciting hands in vertical orientation. Hence we use a vertical rectangle window size as the one we used such as 30x60
 - a classifier for predicting hands in orrizontal orientation. Hence we need an orizontal rectangle window size such as 60x30
 - a classifier for predicting hands in oblique orientation. We might try using a square window size such as 24x24. 

However, given the lack support for gpu computation of the built-in opencv function for training the classifier, training 3 classifier will take some days. Hence, we move towards a more modern feasible solution. 
