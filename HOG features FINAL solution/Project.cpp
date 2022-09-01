#include <iostream>
#include "includes/header.h"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


int main() {

	/*---------------------------------------------------------------
		DATASET set up
	----------------------------------------------------------------*/

	/*
	std::string image_dataset = "../../../../Project/dataset/train/positive";
	std::string image_annotations = "../../../../Project/dataset/train/positive/annotations";
	std::string negativeOriginal = "../../../../Project/dataset/train/background";
	
	createDataset(image_dataset, image_annotations, negativeOriginal);
	*/


	/*---------------------------------------------------------------
		CREATE HOG DETECTORS 
	----------------------------------------------------------------*/

	std::string modelSVMH = "../SVM model/SVMmodelH - 1.xml";
	std::string modelSVMV = "../SVM model/SVMmodelV - 2.xml";

	/*
	createHogDetector(image_dataset + "/rescaled/croppedV", negativeOriginal + "/CroppedV", modelSVMV, windowSizeV);
	createHogDetector(image_dataset + "/rescaled/croppedH", negativeOriginal + "/CroppedH", modelSVMH, windowSizeH);
	*/

	
	/*---------------------------------------------------------------
		DETECTION & INSTANCE SEGMENTATION
	----------------------------------------------------------------*/

	cv::Size windowSizeV = cv::Size(40, 56);	// window size for vertical hands detection
	cv::Size windowSizeH = cv::Size(112, 72);	// window size for horizontal hands detection

	std::string testImages = "../test/rgb";
	std::string annotation = "../test/det";		

	std::vector< cv::Mat > testImg = loadImages(testImages);		// load images
	std::vector< std::string > Annotation = loadTxt(annotation);		// load annotations


	/* DETECTION */
	detectHands(testImg, Annotation, modelSVMH, modelSVMV, windowSizeH, windowSizeV);


	/* INSTANCE SEGMENTATION */
	std::string test2 = "../test/mask";
	std::vector<cv::Mat> maskList = loadImages(test2);
	
	instanceSegmentation(testImg, maskList, modelSVMH, modelSVMV, windowSizeH, windowSizeV);

	return 0;
}



