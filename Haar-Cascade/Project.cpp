#include <iostream>
#include "includes/header.h"
#include <opencv2/highgui.hpp>


int main(){

	/*************************
	    DATASET preparation
	**************************/

	// in the first dataset we convert the polygons points in bounding box (x,y,width,height) storing them in a new xml
	std::string pathDatasetXml1 = "../../../../Project/dataset/hand_over_face/annotations";
	//FromKeypointToBox(pathDatasetXml1);

	// for the second dataset, we convert the original bounding box from the format (Xmin, Xmax, Ymin, Ymax) to the format (x,y,width, height)
	std::string pathDatasetXml2 = "../../../../Project/dataset/EgoHands Public.v1-specific.voc/train";
	//BndBoxConvert(pathDatasetXml2);

	// we convert all the positive samples to grayscale level
	std::string PositiveSamplePath = "../../../../Project/dataset/train/positive";
	//ToGrayscale(PositiveSamplePath);


	/****************************************
	  CREATE FILES FOR TRAINING HAAR CASCADE
	*****************************************/

	// We create the file for positive sample expected to be used for training the cascade
	//createFilePositiveSamples(PositiveSamplePath + "/" + annotations");

	// We create the file for the negative sample expected to be used for training the cascade
	std::string NegativeSamplePath = "../../../../Project/dataset/train/negatives";
	//createFileNegativeSamples(NegativeSamplePath);


	/****************************************
	  DETECT
	*****************************************/
	std::string imagepath = "../../../../Project/dataset/EgoHands Public.v1-specific.voc/test/CHESS_COURTYARD_S_H_frame_2622_jpg.rf.a9a334a54fd5d3a1424b0d387c154341.jpg";
	cv::Mat img = cv::imread(imagepath);
	std::string modelPath = "../../../../Project/dataset/train/haarcascade/cascade.xml";
	cv::Mat out = CascadeDetection(modelPath, img);

	// get the file name of the image
	std::string baseFilename = imagepath.substr(imagepath.find_last_of("/\\") + 1);

	// path where to store the image
	std::string ImageName = "../../../../Project/result/" + baseFilename;

	StoreAndDisplay(out, "output", ImageName);

	return 0;
}
