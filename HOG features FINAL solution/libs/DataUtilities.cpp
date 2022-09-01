#include <iostream>
#include "../includes/header.h"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


/*
* @brief Creates the dataset needed to train the SVM
* @param image_dataset : path of the folder contataining the images for making the positive and negative samples
* @param image_annotations : path of the folder containing the annotation
* @param negPath : path of the image where to store the negative samples
*/
void createDataset(std::string image_dataset, std::string image_annotations, std::string negPath) {
	/*--------------------------------------------------------------
	  0 STEP - DATA PREPROCESS
	--------------------------------------------------------------*/
	dataSetUp(image_dataset, image_annotations);

	/*-------------------------------------------------------------
	  1 STEP : Get the window sizes
	--------------------------------------------------------------*/
	std::string trainSet = image_dataset + "/rescaled";
	std::vector<cv::Size> windows = getWindowsSize(trainSet);

	cv::Size windowSizeV = windows[0];
	cv::Size windowSizeH = windows[1];

	/*-----------------------------------------------------------
	  2 STEP - CREATE THE SAMPLES FOR COMPUTING THE HOG FEATURES
	-------------------------------------------------------------*/
	int padding = 10; // add additional padding to the samples

	// VERTICAL POSITIVE AND NEGATIVE SAMPLES
	std::string negativeDataCroppedV = negPath + "/CroppedV";
	std::string positiveSamplesV = image_dataset + "/croppedV";
	buildSamples(trainSet, positiveSamplesV, negativeDataCroppedV, windowSizeV, padding );

	// HORIZONTAL POSITIVE AND NEGATIVE SAMPLES
	std::string negativeDataCroppedH = negPath + "/CroppedH";
	std::string positiveSamplesH = image_dataset + "/croppedH";
	buildSamples(trainSet, positiveSamplesV, negativeDataCroppedH, windowSizeV, padding);
}


/*
* @brief Utlity function to build the train dataset importing images and annotation from other datasets
* @param image_dataset : The image where to import the images
* @param image_annotations : The image where to import the annotations
* @return It returns the float vector version of the input vector
*/
void dataSetUp(std::string image_dataset, std::string image_annotations) {

	/* --> DATASET: Hand_Over_Face */
	std::string pathDataset1Image = "../../../../Project/dataset/hand_over_face/images_resized";
	std::string pathDataset1Xml = "../../../../Project/dataset/hand_over_face/annotations";

	// we convert polygons annotations to box annotations and copy them to our annotation trian folder
	std::cout << "Converting xml files in progress ... \n";
	FromKeypointToBox(pathDataset1Xml, image_annotations);
	std::cout << " ... DONE \n";

	std::cout << "Copying of images in progress ... \n";
	copyImg(pathDataset1Image, image_dataset);				// we copy all the images to our train folder
	std::cout << " ... DONE \n";


	/* --> DATASET: Ego Hands */
	std::string pathDatasetEgoHand = "../../../../Project/dataset/EgoHands Public.v1-specific.voc/train";

	std::cout << "Copying of images in progress ... \n";
	copyImg(pathDatasetEgoHand, image_dataset);					// copy images to train folder
	std::cout << " ... DONE \n";

	std::cout << "Copying of xml files in progress ... \n";
	copyXml(pathDatasetEgoHand, image_annotations);				// copy xml to annotation train folder
	std::cout << " ... DONE \n";

	// rescale all the train images to width = 600 and height accordingly to keep original ratio*/
	int targetW = 600;
	rescaleKeepRatio(image_dataset, image_annotations, image_dataset, targetW);
}


/**
* @brief Build the positive and negative samples from the same set of images
* @param pathP: The path of the image and annotation from which to create the samples
* @param pathStore: Path where to store the positive samples
* @param negStore: Path where to store the negative sample
* @param winSize: The size to which to resize the samples
* @param padding: Additional padding to add to the size of the positive samples for making them larger
*/
void buildSamples(std::string pathP, std::string pathStore, std::string negStore, cv::Size winSize, int padding) {

	// POSITIVE SAMPLES
	createSamples(pathP, pathP, pathStore, winSize.width, winSize.height, padding);

	/* Optionally augment the data */
	//augmentData(pathStore);

	// NEGATIVE SAMPLES
	createNegativeHogSamplesFromPositive(pathP, pathP, negStore, winSize.width, winSize.height);
}

/**
* @brief Build the negative samples from the positve images
* @param positiveImages: The path of the image from which to create the negative samples
* @param positiveAnnotations: The annotations of the groun boxes
* @param path: The path where to store the negative samples
* @param targetW: The width that the negative samples need to have
* @param targetH: The height that the negative samples need to have
* #return It creaates and stores the negative samples
*/
void createNegativeHogSamplesFromPositive(std::string positiveImages, std::string positiveAnnotations, std::string path, int targetW, int targetH) {

	std::vector<cv::String> xmlList = loadXmls(positiveAnnotations);
	std::vector<cv::String> imageList = loadImgs(positiveImages);

	// reorder list
	sort(imageList.begin(), imageList.end());
	sort(xmlList.begin(), xmlList.end());
	for (int j = 0; j < 288; j = j + 1) {

		cv::Mat img = cv::imread(imageList[j]);

		// get name of the image
		std::string base_filename = imageList[j].substr(imageList[j].find_last_of("/\\") + 1);
		std::string imgFileName = base_filename.erase(base_filename.length() - 4); // we remove the '.jpg' from the filename

		// get bounding boxes
		std::vector<cv::Rect> boxes = getTruthBoxes(xmlList[j]);

		// get random sample background
		cv::Size imageSize = img.size();

		for (int z = 0; z < (imageSize.width - targetW); z = z + targetW) {

			for (int i = 0; i < (imageSize.height - targetH); i = i + targetH) {

				cv::Rect backgr = cv::Rect(z, i, targetW, targetH);

				bool feasible = true;

				// for each ground truth boxes, we check it doesn't contain the obj
				for (int k = 0; k < boxes.size(); k++) {

					double IOU = iou(backgr, boxes[k]);

					// if found one, crop the image along it
					if (IOU != 0.0) {
						feasible = false;
					}
				}

				// check if the box is containing an object of interest
				if (feasible) {
					cv::Mat croppedImg = img(backgr).clone();

					// convert to grayscale
					cv::Mat gray;
					cvtColor(croppedImg, gray, cv::COLOR_BGR2GRAY);

					// equalize histogram
					cv::Mat out = gray;
					cv::equalizeHist(gray, out);

					// apply bilateral filter for preserving contours and remove noise
					cv::Mat denoised;
					cv::bilateralFilter(out, denoised, 3, 20, 20);

					// store it
					cv::imwrite(path + "/" + imgFileName + " - " + std::to_string(i) + std::to_string(z) + ".jpg", denoised);
				}

			}
		}

	}

}


/**
* @brief Create the positive samples
* @param positiveImages: The path of the folder containing the images from which to create the positive samples
* @param positiveAnnotations: The path of the folder contatining the annotations
* @param path: The path of the folder where to store the samples
* @param targetW: The width that the positive samples need to have
* @param targetH: The height that the positive samples need to have
* @param padding: Additional padding to add to the size of the samples for making them larger
* #return It creaates and stores the positive samples
*/
void createSamples(std::string positiveImages, std::string positiveAnnotations, std::string path, int targetW, int targetH, int padding) {

	std::vector<cv::String> xmlList = loadXmls(positiveAnnotations);
	std::vector<cv::String> imageList = loadImgs(positiveImages);

	// reorder list
	sort(imageList.begin(), imageList.end());
	sort(xmlList.begin(), xmlList.end());

	for (int j = 0; j < imageList.size(); j++) {

		cv::Mat img = cv::imread(imageList[j]);

		// get name of the image
		std::string base_filename = imageList[j].substr(imageList[j].find_last_of("/\\") + 1);
		std::string imgFileName = base_filename.erase(base_filename.length() - 4); // we remove the '.jpg' from the filename

		// get horizontal bounding boxes
		std::vector<cv::Rect> boxes = getHorizontalBoxes(xmlList[j]);

		for (int i = 0; i < boxes.size(); i++) {

			// pad a bit the roi
			boxes[i].x = std::max(boxes[i].x - padding, 0);
			boxes[i].y = std::max(boxes[i].y - padding, 0);
			boxes[i].width = std::min(boxes[i].width + padding, img.cols - boxes[i].x);
			boxes[i].height = std::min(boxes[i].height + padding, img.rows - boxes[i].y);

			// cropped along the rectangle
			cv::Mat croppedImg = img(boxes[i]).clone();

			// convert to grayscale
			cv::Mat gray;
			cvtColor(croppedImg, gray, cv::COLOR_BGR2GRAY);

			// equalize histogram
			cv::Mat out = gray;
			cv::equalizeHist(gray, out);

			// rescale
			cv::Mat resized;
			cv::resize(out, resized, cv::Size(targetW, targetH), cv::INTER_CUBIC);

			// apply bilateral filter for preserving contours and remove noise
			cv::Mat denoised;
			cv::bilateralFilter(resized, denoised, 5, 50, 50);

			// store it
			cv::imwrite(path + "/" + imgFileName + " - " + std::to_string(i) + ".jpg", denoised);

		}

	}

}



/**
* @brief Augment the data
* @param path: The path of the folder where the images are stored. Here the new augmented images will be stored.
* #return It augments the data
*/
void augmentData(std::string path) {

	std::vector<cv::String> imageList = loadImgs(path);

	// reorder list
	sort(imageList.begin(), imageList.end());

	std::cout << imageList.size();

	for (int j = 0; j < imageList.size(); j = j + 2) {

		cv::Mat img = cv::imread(imageList[j]);

		std::string base_filename = imageList[j].substr(imageList[j].find_last_of("/\\") + 1);
		std::string imgFileName = base_filename.erase(base_filename.length() - 4); // we remove the '.jpg' from the filename

		// flip around y
		cv::Mat horizontal_flip;
		cv::flip(img, horizontal_flip, 1);

		// store it
		cv::imwrite(path + "/" + imgFileName + " - H" + std::to_string(j) + ".jpg", horizontal_flip);

		// flip around x
		cv::Mat vertical_flip;
		cv::flip(img, vertical_flip, 0);

		// store it
		cv::imwrite(path + "/" + imgFileName + " - V" + std::to_string(j) + ".jpg", vertical_flip);

		// flip around both axis
		cv::Mat both_flip;
		cv::flip(img, both_flip, -1);

		// store it
		cv::imwrite(path + "/" + imgFileName + " - VH" + std::to_string(j) + ".jpg", both_flip);
	}
}