# Hand-detection-and-segmentation
Here we present our solution for the Hand detection and segmentation project for the Computer Vision course.

Our first approach was based on using an Haar cascade Classifier for handling the detection task. However we moved to a different solution since the Haar cascade takes too long for getting trained properly in order to get good results. 

Our final solution is based on using HOG features + SVM + a skin-detector for managing the detection of the hands. The segmentation part has instead been carried out by exploiting the grabcut algorithm. Please read the report inside the 'HOG feaures final solution' for more details about our implementation.
 
