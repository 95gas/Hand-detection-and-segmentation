#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>  // used for remove last n character 
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>	// for detection with cascade classifier
#include <opencv2/dnn/dnn.hpp>		// for applying the non-maxima suppression over the boxes

#include "../includes/header.h"

cv::Mat CascadeDetection(std::string modelPath, cv::Mat img) {
	
	// convert image to grayscale
	cv::Mat gray;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);

	// equalize Histogram for enhancing the contrast
	cv::equalizeHist(gray, gray);

	/**********************
		OBJECT DETECTION
	**********************/

	// Load model
	cv::CascadeClassifier model;
	if (!model.load(modelPath)) {
		std::cout << "ERROR: model couldn't be loaded!";
		exit(EXIT_FAILURE);
	}

	std::vector<int> levels;
	std::vector<double> scores;
	std::vector<cv::Rect> hands;

	model.detectMultiScale(gray, hands, levels, scores, 1.1, 3, 0, cv::Size(), cv::Size(), true); 

	// check if no hands have been detected
	if (hands.size() == 0) {
		std::cout << "No hands founds in the image";
		exit(EXIT_SUCCESS);
	}

	// Non maxima suppression

	float scoreThreshold = 0.8;
	float nmsThreshold = 0.5;
	std::vector<int> index;
	std::vector<float> scoresFloat = toFloatVec(scores);  	// convert double to float

	cv::dnn::NMSBoxes(hands, scoresFloat, scoreThreshold, nmsThreshold, index, 1.f, 0);

	// draw bounding boxes over image
	for (int i = 0; i < index.size(); i++) {
		int j = index[i];
		cv::Point p1 = cv::Point(hands[j].x, hands[j].y);
		cv::Point p2 = cv::Point(hands[j].x + hands[j].width, hands[j].y + hands[j].height);

		cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 0));
	}

	return img;
}


/**
* @brief Utlity functio to convert a double vector to a float vector
* @param vec: The double vector to convert
* @return It returns the float vector version of the input vector
*/
std::vector<float> toFloatVec(std::vector<double> vec) {

	std::vector<float> result;

	for (int i = 0; i < vec.size(); i++) {
		float num = (float)vec[i];
		result.push_back(num);
	}

	return result;
}


/**
* @brief Store and display the image. The image is store as png image
* @param img: image
* @param NameWindow: name of the window on which it will be displayed the image
* @param NameImage: path where to store the image
*/
void StoreAndDisplay(cv::Mat img, std::string NameWindow, std::string NameImage) {

	// store results
	cv::imwrite(NameImage, img);

	// display result
	cv::imshow(NameWindow, img);
	cv::waitKey(0);
}