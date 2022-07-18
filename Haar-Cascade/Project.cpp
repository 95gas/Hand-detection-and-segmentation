// Project.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "includes/header.h"
#include <opencv2/highgui.hpp>


int main(){

	/*************************
	* DATASET preparation
	***************************/

	std::string pathDataset = "../../../../Project/dataset/hand_over_face/annotations";

	//FromKeypointToBox(pathDataset);

	//BndBoxConvert("../../../../Project/dataset/EgoHands Public.v1-specific.voc/valid");

	//ToGrayscale("../../../../Project/dataset/train/positive");

	// let's take 1000 positive samples. We create the file expected to be used for training the cascade
	//createFilePositiveSamples("../../../../Project/dataset/train/positive/annotations");

	// let't take 3000 negative samples
	//createFileNegativeSamples("../../../../Project/dataset/train/negatives");

	return 0;
}
