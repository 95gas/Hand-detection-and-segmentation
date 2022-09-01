#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core_detect.hpp>
#include <string>							// used for remove last n character 
#include <vector>
#include <opencv2/highgui.hpp>
#include <fstream>							// for creating file
#include <math.h>							// for floor function

#include "../includes/header.h"
#include "../includes/rapidxml/rapidxml.hpp" // For parsing the xml
#include "../includes/tinyxml2.h"		     // For creating the xml files
#include <filesystem>
#include <opencv2/objdetect.hpp>			// for HOG features computation
#include <opencv2/dnn/dnn.hpp>


/**
* @brief create an HOG-based detector
* @param positiveImages: The path of the folder containing the positve samples
* @param negativeDataCropped: The path of the folder containing the negative samples
* @param pathModel: The path where to store the file of the SVM model
* @param winsize: The size of the window
* #return It creates the file of the SVM and stores it in the pathModel folder
*/
void createHogDetector(std::string positiveImages, std::string negativeDataCropped, std::string pathModel, cv::Size winsize) {

	/*--------------------------------------------------------------------------
	         COMPUTE HOG FEATURES for positive and negative samples
	---------------------------------------------------------------------------*/
	std::vector<cv::String> positiveSamplesV = loadImgs(positiveImages);
	std::vector<cv::String> negativeSamplesV = loadImgs(negativeDataCropped);

	std::vector< int > labels;							

	std::cout << "\nHistogram of Gradients are being calculated for positive images...\n";
	std::vector<cv::Mat> gradientP = computeHog(winsize, positiveSamplesV);

	int positive_count = gradientP.size();
	labels.assign(positive_count, +1);						// assign class +1 (=hands) to our positive samples

	std::cout << "...[done] ( positive images count : " << positive_count << " )\n";

	std::cout << "\nHistogram of Gradients are being calculated for negative images...\n";
	std::vector<cv::Mat> gradientN = computeHog(winsize, negativeSamplesV);
	std::vector<cv::Mat> gradient_lst = gradientP;

	for (int i = 0; i < gradientN.size(); i++) {
		gradient_lst.push_back(gradientN[i]);
	}

	int negative_count = gradient_lst.size() - positive_count;
	labels.insert(labels.end(), negative_count, -1);		// assign class -1 (= no hands) to our negative samples
	std::cout << "...[done] ( negative images count : " << negative_count << " )\n";


	/*-----------------------------------------------------------------------------
				TRAIN THE SVM with Negative Mining and store it
	-------------------------------------------------------------------------------*/

	trainSVM(gradient_lst, labels, pathModel, winsize);
}



/**
* @brief Compute the horizontal and vertical windows
* @param positiveAnnotations: The path of the folder contatining the annotations
* #return It returns the windows size for the horizontal and vertical window
*/
std::vector<cv::Size> getWindowsSize(std::string positiveAnnotations) {

	std::vector<cv::String> xmlList = loadXmls(positiveAnnotations);

	std::cout << "Getting window sizes .. ";
	std::vector<cv::Rect> Vboxes = getAllVerticalBoxes(xmlList);
	std::vector<cv::Rect> Hboxes = getAllHorizontalBoxes(xmlList);

	cv::Size aveSizeV = getAverageBoxSize(Vboxes);
	std::cout << "\n -> The average size of vertical boxes : ( width : " + std::to_string(aveSizeV.width) + ", height : " + std::to_string(aveSizeV.height) + ")\n";

	cv::Size aveSizeH = getAverageBoxSize(Hboxes);
	std::cout << " -> The average size of horizontal boxes : ( width : " + std::to_string(aveSizeH.width) + ", height : " + std::to_string(aveSizeH.height) + ")\n";


	/* Define the windows size to be the closest integer multiple of the pixels per cell */
	int pixels_per_cell = 8;

	/* VERTICAL WINDOW */
	int tmp = (floor(aveSizeV.height) /2)/ pixels_per_cell;	// second experiment
	int tmp1 = (floor(aveSizeV.width) /2)/ pixels_per_cell;	// second experiment
	//int tmp = floor(aveSizeV.height) / pixels_per_cell;	// first experiment
	//int tmp1 =floor(aveSizeV.width) / pixels_per_cell;	// first experiment
	int heightV = (tmp + 1) * pixels_per_cell;
	int widthV = (tmp1 + 1) * pixels_per_cell;
	cv::Size windowSizeV = cv::Size(widthV, heightV);
	std::cout << " -> The window size of vertical boxes : ( width : " + std::to_string(windowSizeV.width) + ", height : " + std::to_string(windowSizeV.height) + ")\n";

	/* HORIZONTAL WINDOW */
	tmp = (floor(aveSizeH.height) / 2) / pixels_per_cell;	// second experiment
	tmp1 = (floor(aveSizeH.width) / 2) / pixels_per_cell;	// second experiment
	//tmp = floor(aveSizeH.height) / pixels_per_cell;		// first experiment
	//tmp1 = floor(aveSizeH.width) / pixels_per_cell;		// first experiment
	int heightH = (tmp + 1) * pixels_per_cell;
	int widthH = (tmp1 + 1) * pixels_per_cell;
	cv::Size windowSizeH = cv::Size(widthH, heightH);
	std::cout << " -> The window size of horizontal boxes : ( width : " + std::to_string(windowSizeH.width) + ", height : " + std::to_string(windowSizeH.height) + ")\n";

	std::vector<cv::Size> result;
	result.push_back(windowSizeV);
	result.push_back(windowSizeH);

	return result;
}



/**
* @brief Train the SVM over the HOG features
* @param gradient_lst: The HOG features of both the positive and negative samples
* @param labels: The class labels associated to the HOG features
* @param path: The path where to store the SVM model
* @param winsize: The window size of the detector to train
* #return It returns the windows size for the horizontal and vertical window
*/
void trainSVM(std::vector<cv::Mat> gradient_lst, std::vector< int > labels, std::string path, cv::Size winsize) {
	
	// convert data in machine learning format for the SVM
	cv::Mat train_data;
	convert_to_ml(gradient_lst, train_data);	

	std::cout << "\nTraining SVM...";
	// Default values to train SVM 
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-5));
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setP(0.2); 
	svm->setC(100); 
	svm->setType(cv::ml::SVM::EPS_SVR); 
	svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
	std::cout << "...[done]\n";

	cv::Size blockSize = cv::Size(16, 16);
	cv::Size blockStride = cv::Size(4, 4);
	cv::Size cellSize = cv::Size(8, 8);
	int nbins = 9;

	// store SVM model
	cv::HOGDescriptor hog = cv::HOGDescriptor(winsize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(get_svm_detector(svm));
	hog.save(path);
	
}





/**
* @brief Compute the HOG features
* @param wsize: The window size
* @param img_lst: The list of the path of the images of which to compute the HOG features
* #return It returns the HOG features for all the images given in input
*/
std::vector<cv::Mat>  computeHog(cv::Size wsize, std::vector<cv::String> img_lst) {

	std::vector<cv::Mat> gradient_lst;

	// HOG features parameters
	cv::Size blockSize = cv::Size(16, 16);
	cv::Size blockStride = cv::Size(4, 4);
	cv::Size cellSize = cv::Size(8, 8);
	int nbins = 9;

	cv::HOGDescriptor hog = cv::HOGDescriptor(wsize, blockSize, blockStride, cellSize, nbins);

	// hog.block size is set to (16x16) as it is the only one supported

	std::vector<float> descriptors;

	for (int i = 0; i < img_lst.size(); i++)
	{
		cv::Mat img = cv::imread(img_lst[i]);

		if (img.cols >= wsize.width && img.rows >= wsize.height)
		{
			hog.compute(img, descriptors, blockStride, cv::Size(0, 0));
			gradient_lst.push_back(cv::Mat(descriptors).clone());
			
		}
	}

	return gradient_lst;
}


/*
* @brief Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* @train_samples: the input data to convert
* @TrainData: is a matrix of size (#samples x max(#cols,#rows) per samples in 32FC1. This is where the output is stored
*/
void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	for (int i = 0; i < train_samples.size(); ++i)
	{
		if (train_samples[i].cols == 1)
		{
			transpose(train_samples[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (train_samples[i].rows == 1)
		{
			train_samples[i].copyTo(trainData.row((int)i));
		}
	}
}



/*
* @brief Get the SVM for the detector
* @svm the SVM 
* @return It returns the SVM function necessary to perform the classification
*/
std::vector< float > get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm)
{
	// get the support vectors
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	std::vector< float > hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float) -rho;
	return hog_detector;
}



/*
* @brief Perform the detection using the SVM model over a pyramid scale space
* @SVM_ModeFile The SVM model file
* #img The image to analyse
* @winsize The size of the sliding windows
* @finalBox The predicted boxes list
* @finalScores The confidential score list
* @scoreThreshold The score threshold to apply for the no max suppression
* @nmsThreshold The IoU threshold to apply for the no max suppression
* @return It returns the SVM function necessary to perform the classification
*/
void SVMdetector(std::string SVM_ModeFile, cv::Mat img, cv::Size winsize, std::vector< cv::Rect >& finalBox, std::vector<float>& finalScores, float scoreThreshold, float nmsThreshold)
{
	// HOG features parameters
	cv::Size blockSize = cv::Size(16, 16);
	cv::Size blockStride = cv::Size(4, 4);
	cv::Size cellSize = cv::Size(8, 8);
	int nbins = 9;

	// load SVM
	cv::HOGDescriptor hog = cv::HOGDescriptor(winsize, blockSize, blockStride, cellSize, nbins);
	hog.load(SVM_ModeFile);

	if (img.empty())
	{
		return;
	}

	// perform detection in the pyramid scale space with windows stride (16,16)
	std::vector< cv::Rect > detections;
	std::vector< double > foundWeights;

	hog.detectMultiScale(img, detections, foundWeights, 0.0, cv::Size(16,16), cv::Size(8, 8), 1.01, 0, false); // small 1.05


	if (detections.size() != 0) {	// if found something
		
		
		// perform no-maxima suppression
		std::vector<int> index;
		std::vector<float> scoresFloat = toFloatVec(foundWeights);  	// convert double to float

		cv::dnn::NMSBoxes(detections, scoresFloat, scoreThreshold, nmsThreshold, index, 1.f, 10);

		// updates the list of predicted boxes
		if (index.size() != 0) {

			for (int i = 0; i < index.size(); i++) {
				int j = index[i];
				finalBox.push_back(detections[j]);
				finalScores.push_back(scoresFloat[j]);
			}

			//cv::imshow("Detection", img);
			//cv::waitKey(0);
		}
		else {
			//std::cout << "No hands present in the image!";
		}
	}
	else {
		//std::cout << "No hands present in the image!";
	}
}



/*
* @brief Perform the detection of two SVMs by applying no maxima suppression on the combination of the predicted boxes by SVM1 and SVM2, and filter out the resulting boxes based on the skin detector and the area threshold
* #img The image to analyse
* @SVM1 The file of the first SMV model
* @SVM2 The file of the second SMV model
* @winsize1 The windows size of SVM1
* @winsize2 The window size of SVM2
* @return It returns the predicted boxes resulting by applying no maxima suppression over the combination of the predicted boxes by SVM1 and SVM2
*/
std::vector<cv::Rect> detectBox(cv::Mat img, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2) {

	/* get boxes from SVM 1 */
	std::vector< cv::Rect > boxes1;
	std::vector<float> scores1;
	SVMdetector(SVM1, img, winsize1, boxes1, scores1, 0.6,0.1);
	//test_trained_detector(SVM1, img, winsize1, boxes1, scores1, 0.82, 0.1); // second experiment
	//test_trained_detector(SVM1, img, winsize1, boxes1, scores1, 0.6, 0.1);  // first experiment


	/* get boxes from SVM 2 */
	std::vector< cv::Rect > boxes2;
	std::vector<float> scores2;
	SVMdetector(SVM2, img, winsize2, boxes2, scores2,0.6,0.1);
	//test_trained_detector(SVM2, img, winsize2, boxes2, scores2, 0.82, 0.1); // second experiment
	//test_trained_detector(SVM2, img, winsize2, boxes2, scores2, 0.6, 0.1);  // first experiment

	
	// Combine boxes
	std::vector<cv::Rect> finalBox;
	std::vector<float> finalScores;
	std::vector<int> finalIndex;

	finalBox = boxes1;
	finalScores = scores1;

	for (int i = 0; i < boxes2.size(); i++) {

		finalBox.push_back(boxes2[i]);
		finalScores.push_back(scores2[i]);
	}

	// Non maxima suppression 
	float scoreThreshold = 0.75; 	// experiment1: 0.75  // experiment2: 0.7
	float nmsThreshold = 0.1; 		// experiment1: 0.08  // experiment2: 0.1
	cv::dnn::NMSBoxes(finalBox, finalScores, scoreThreshold, nmsThreshold, finalIndex, 1.f, 4);

	std::vector<cv::Rect> prediction;

	for (int i = 0; i < finalIndex.size(); i++) {
		int j = finalIndex[i];
		prediction.push_back(finalBox[j]);
	}

	/*Filter out the predicted boxes based on the skin detector and the area threshold*/
	std::vector<cv::Rect> output;
	int AreaTH = 25000;

	for (int k = 0; k < prediction.size(); k++) {
		if (skinDetector(img(prediction[k]))) {
			int bndArea = prediction[k].width * prediction[k].height;
			if (bndArea < AreaTH) {
				output.push_back(prediction[k]);
			}
		}
	}

	return output;
}


/*
* @brief It performs the countour-based instance segmentation technique
* @img The img to segment
* @prediction The list of the ROI where to look for performing the segmentation
* @return It returns the mask of the instance segmentation
*/
cv::Mat InstanceSegContour(cv::Mat img, std::vector<cv::Rect> prediction) {

	// Parameters
	int canny_low = 15;
	int canny_high = 200;
	int dilate_iter = 10;
	int erode_iter = 10;

	cv::Mat finalMask = cv::Mat::zeros(img.rows, img.cols, CV_8U);

	// initialize random number generato. We used it for generating random colour
	cv::RNG rng(0xFFFFFFFF);		

	for (int j = 0; j < prediction.size(); j++) {

		// take ROI
		cv::Mat bnd = img(prediction[j]);

		// convert to grayscale
		cv::Mat gray;
		cv::cvtColor(bnd, gray, cv::COLOR_BGR2GRAY);

		// Apply Canny Edge Dection
		cv::Mat edges = bnd.clone();
		cv::Canny(gray, edges, canny_low, canny_high);

		// Morphologic operations
		cv::dilate(edges, edges, cv::Mat(), cv::Point(-1,-1), dilate_iter);
		cv::erode(edges, edges, cv::Mat(), cv::Point(-1, -1), erode_iter);

		// find contours
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		// set up mask with a matrix of zero
		cv::Mat mask = cv::Mat::zeros(edges.rows, edges.cols, CV_8U);

		// find relevant contours and add to mask
		for (int k = 0; k < contours.size(); k++) {
			// we add the foreground object to the mask
			cv::fillConvexPoly(mask, contours[k], 255);
		}

		// use dilate, erode, and blur to smooth out the mask
		cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), dilate_iter);
		cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), erode_iter);
		cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);

		// segment object on image
		int x_mask = 0;
		int icolor = (unsigned)rng;

		for (int x = prediction[j].x; x < prediction[j].x + prediction[j].width; x++) {
			int y_mask = 0;

			for (int y = prediction[j].y; y < prediction[j].y + prediction[j].height; y++) {

				// colour foreground pixels
				if (mask.at<unsigned char>(y_mask, x_mask) == 255) {
					img.at<cv::Vec3b>(y, x) = cv::Vec3b(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
					finalMask.at<unsigned char>(y, x) = 255;
				}
				y_mask++;
			}
			x_mask++;
		}
	}

	cv::imshow("Instance segmentation", img);
	cv::waitKey(0);

	return finalMask;
}


/*
* @brief It performs the Grabcut instance segmentation technique
* @img The img to segment
* @prediction The list of the ROI where to look for performing the segmentation
* @return It returns the mask of the instance segmentation
*/
cv::Mat instaSegGrabcut(cv::Mat img, std::vector<cv::Rect> prediction) {

	cv::RNG rng(0xFFFFFFFF);	// 	initialize random number generato. We used it for generating random colour

	cv::Mat finalMask = cv::Mat::zeros(img.rows, img.cols, CV_8U);

	for (int j = 0; j < prediction.size(); j++) {

		// perform Grabcut method
		cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		cv::Mat bgdModel;
		cv::Mat fgdModel;	
		cv::grabCut(img, mask, prediction[j], bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);

		for (int x = 0; x < mask.cols; x++) {
			for (int y = 0; y < mask.rows; y++) {

				// set background pixels
				if (mask.at<unsigned char>(y, x) == 2) {
					finalMask.at<unsigned char>(y, x) = 0;
					mask.at<unsigned char>(y, x) = 0;
				}

				// set foreground pixels
				if (mask.at<unsigned char>(y, x) == 3) {
					finalMask.at<unsigned char>(y, x) = 255;
					mask.at<unsigned char>(y, x) = 255;
				}

				// set foreground pixels
				if (mask.at<unsigned char>(y, x) == 1) {
					finalMask.at<unsigned char>(y, x) = 255;
					mask.at<unsigned char>(y, x) = 255;
				}
			}
		}

		int icolor = (unsigned)rng; 		// generate random color

		// colour foreground object in the image with random colour
		for (int x = 0; x < mask.cols; x++) {
			for (int y = 0; y < mask.rows; y++) {
				if (mask.at<unsigned char>(y, x) == 255) {
					img.at<cv::Vec3b>(y, x) = cv::Vec3b(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
				}
			}
		}
	}
	cv::imshow("Instance segmentation - GrabCut", img);
	cv::waitKey(0);

	return finalMask;
}




/*
* @brief It performs the colour thresholding instance segmentation technique
* @img The img to segment
* @prediction The list of the ROI where to look for performing the segmentation
* @return It returns the mask of the instance segmentation
*/
cv::Mat instaTH(cv::Mat img, std::vector<cv::Rect> prediction) {

	cv::Mat finalMask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
	int dilate_iter = 10;
	int erode_iter = 10;

	cv::RNG rng(0xFFFFFFFF); 	// initialize random number generato. We used it for generating random colour

	for (int j = 0; j < prediction.size(); j++) {

		// take ROI
		cv::Mat bnd = img(prediction[j]);

		int AreaTH = 25000;
		int bndArea = bnd.rows * bnd.cols;

		if (bndArea < AreaTH) {
			cv::Mat hsv;
			cv::cvtColor(bnd, hsv, cv::COLOR_RGB2HSV);

			cv::Mat ycb;
			cv::cvtColor(bnd, ycb, cv::COLOR_RGB2YCrCb);

			cv::Mat bndA;
			cv::cvtColor(bnd, bndA, cv::COLOR_RGB2RGBA);

			// set up mask with a matrix of zero
			cv::Mat mask = cv::Mat::zeros(bnd.rows, bnd.cols, CV_8U);

			// check if pixel is an hand pixel
			for (int col = 0; col < bnd.cols; col++) {
				for (int row = 0; row < bnd.rows; row++) {
					int R = bndA.at<cv::Vec4b>(row, col)[0];
					int G = bndA.at<cv::Vec4b>(row, col)[1];
					int B = bndA.at<cv::Vec4b>(row, col)[2];
					int A = bndA.at<cv::Vec4b>(row, col)[3];

					// check in RGB colour space
					if (R > 95 && G > 40 && B > 20 && R > G && R > B && abs(R - G) > 15 && A > 15) {

						// mark pixel as hands
						mask.at<unsigned char>(row, col) = 255;
					}

					// check in the HSV color space
					double H = hsv.at<cv::Vec3b>(row, col)[0];
					double S = hsv.at<cv::Vec3b>(row, col)[1];
					double V = hsv.at<cv::Vec3b>(row, col)[2];
					if (H <= 100 && (S >= 0.23 * 255 || S <= 0.68 * 255)) {

						// mark pixel as hands
						mask.at<unsigned char>(row, col) = 255;
					}

					// check in the YCbCr color space
					double Y = ycb.at<cv::Vec3b>(row, col)[0];
					double Cr = ycb.at<cv::Vec3b>(row, col)[1];
					double Cb = ycb.at<cv::Vec3b>(row, col)[2];

					if (Cr > 135 && Cb > 85 && Y > 80) {
						if ((Cr <= (1.5862 * Cb) + 20) && (Cr >= (0.3448 * Cb) + 76.2069)) {
							if ((Cr >= (-4.5652 * Cb) + 234.5652) && (Cr <= (-1.15 * Cb) + 301.75) && (Cr <= (-2.2857 * Cb) + 432.85)) {

								// mark pixel as hand
								mask.at<unsigned char>(row, col) = 255;
							}
						}
					}
				}
			}

			// use dilate, erode, and blur to smooth out the mask
			cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), dilate_iter);
			cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), erode_iter);
			cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);

			// segment object in image
			int x_mask = 0;
			int icolor = (unsigned)rng;

			for (int x = prediction[j].x; x < prediction[j].x + prediction[j].width; x++) {
				int y_mask = 0;

				for (int y = prediction[j].y; y < prediction[j].y + prediction[j].height; y++) {

					// colour foreground pixels with random colours
					if (mask.at<unsigned char>(y_mask, x_mask) == 0) {
						img.at<cv::Vec3b>(y, x) = cv::Vec3b(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
						finalMask.at<unsigned char>(y, x) = 255;
					}
					y_mask++;
				}
				x_mask++;
			}
		}	
	}

	cv::imshow("Instance segmentation with Skin detector", img);
	cv::waitKey(0);

	return finalMask;
}


/*
* @brief It implements the skin detector
* @img The img to classify
* @return It returns a boolean value
*/
bool skinDetector(cv::Mat img) {
	int count = 0;

	cv::Mat hsv;
	cv::cvtColor(img, hsv, cv::COLOR_RGB2HSV);

	cv::Mat ycb;
	cv::cvtColor(img, ycb, cv::COLOR_RGB2YCrCb);

	cv::Mat bndA;
	cv::cvtColor(img, bndA, cv::COLOR_RGB2RGBA);


	// check if pixel is an hand pixel
	for (int col = 0; col < img.cols; col++) {
		for (int row = 0; row < img.rows; row++) {
			int R = bndA.at<cv::Vec4b>(row, col)[0];
			int G = bndA.at<cv::Vec4b>(row, col)[1];
			int B = bndA.at<cv::Vec4b>(row, col)[2];
			int A = bndA.at<cv::Vec4b>(row, col)[3];

			// check in the RGB colour space
			if (R > 95 && G > 40 && B > 20 && R > G && R > B && abs(R - G) > 15 && A > 15) {

				count++;
			}

			// check in the HSV color space
			double H = hsv.at<cv::Vec3b>(row, col)[0];
			double S = hsv.at<cv::Vec3b>(row, col)[1];
			double V = hsv.at<cv::Vec3b>(row, col)[2];
			if (H <= 100 && (S >= 0.23 * 255 || S <= 0.68 * 255)) {
				count++;
			}

			// check in the YCbCr color space
			double Y = ycb.at<cv::Vec3b>(row, col)[0];
			double Cr = ycb.at<cv::Vec3b>(row, col)[1];
			double Cb = ycb.at<cv::Vec3b>(row, col)[2];

			if (Cr > 135 && Cb > 85 && Y > 80) {
				if ((Cr <= (1.5862 * Cb) + 20) && (Cr >= (0.3448 * Cb) + 76.2069)) {
					if ((Cr >= (-4.5652 * Cb) + 234.5652) && (Cr <= (-1.15 * Cb) + 301.75) && (Cr <= (-2.2857 * Cb) + 432.85)) {
						count++;
					}
				}
			}
		}
	}

	// classify image based on the number of skin pixels detected
	if ((img.rows* img.cols) - count > 700){
		return true;
	}
	return false;
}



/*
* @brief Detect the hands present in a set of images and shows the IoU of the predicted boxes
* @images The list of the images over which to detect hands
* @annotations The list of the path where the annotation files are
* @SVM1 SVM model for horizontal hands detection
* @SMM2 SVM model for vertical hands detection
* @winsize1 window size for horizontal hands detection
* @winsize2 window size for vertical hands detection
* @return It shows on a window the predicted boxes with the corresponding IoU
*/
void detectHands(std::vector< cv::Mat > images, std::vector< std::string > annotations, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2) {
		for (int i = 0; i < images.size(); i++) {

			// IOU
			std::vector<cv::Rect> groundBoxes = getBoxesFromTxt(annotations[i]);
			int targetWidth = 600;

			// check image size
			if (images[i].rows != targetWidth) {

				cv::Size imageSizeOld = images[i].size();

				images[i] = rescaleImg(images[i], targetWidth);
				cv::Size imageSizeNew = images[i].size();

				// rescale boxes
				groundBoxes = rescaleB(groundBoxes, imageSizeOld, imageSizeNew);
			}

			std::vector<cv::Rect> predBoxes = detectBox(images[i], SVM1, SVM2, winsize1, winsize2);

			// compute performance by IoU
			std::vector<double> iouScore = IoU(groundBoxes, predBoxes);

			std::cout << " \nBest IoU for each ground boxes in image " + std::to_string(i) + " [ tot ground boxes :" + std::to_string(groundBoxes.size()) + " ]\n";
			if (predBoxes.size() == 0) {
				std::cout << " -> No hands found in the image!\n";
			}
			else {
				for (int j = 0; j < iouScore.size(); j++) {
					std::cout << " -> " + std::to_string(iouScore[j]) + "\n";
				}
			}

			// draw bounding boxes over image
			cv::Mat tmp = images[i].clone();
			for (int k = 0; k < predBoxes.size(); k++) {
				cv::rectangle(tmp, predBoxes[k], cv::Scalar(0, 255, 0));

				// wrte IoU on the predicted box over the image
				cv::Point bottom_left_corner = cv::Point(predBoxes[k].x, predBoxes[k].y - 5);
				std::string val = std::to_string(iouScore[k]);		// trunk up to 2 decimals
				std::string trunk = val.substr(0, 4);
				cv::putText(tmp, trunk, bottom_left_corner, cv::FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0, 255, 0));
			}

			//cv::imwrite("../../../../Project/dataset/test/result/" + std::to_string(i) + "- det.jpg", tmp);
			cv::imshow("Detection", tmp);
			cv::waitKey(0);
		}
		cv::destroyWindow("Detection");
}


/*
* @brief Perform instance segmenation of hands using the Grabcut method. It also computes the pixel accuracy
* @images The list of the images over which to detect hands
* @gtMasks The list of the ground truth masks
* @SVM1 SVM model for horizontal hands detection
* @SMM2 SVM model for vertical hands detection
* @winsize1 window size for horizontal hands detection
* @winsize2 window size for vertical hands detection
* @return It shows on a window the instance segmentation results and prints at standard output the corresponding pixel accuracy
*/
void instanceSegmentation(std::vector< cv::Mat > images, std::vector< cv::Mat > gtMasks, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2) {
	
	for (int i = 0; i < gtMasks.size(); i++) {

		// check image and mask size
		int targetWidth = 600;
		if (gtMasks[i].rows != targetWidth) {
			gtMasks[i] = rescaleImg(images[i], 600);
			images[i] = rescaleImg(images[i], targetWidth);
		}

		// get boxes
		cv::Mat tmp = images[i].clone();
		std::vector<cv::Rect> predBoxes = detectBox(images[i], SVM1, SVM2, winsize1, winsize2);

		// get mask and perform instance segmentation
		cv::Mat predMask = instaSegGrabcut(tmp, predBoxes);
		//cv::imwrite("../../../../Project/dataset/test/result/" + std::to_string(i) + "- Contour - seq.jpg", tmp);

		// compute Pixel accuracy
		cv::Mat bgMask = gtMasks[i];
		double accuracy = pixelAcc(bgMask, predMask);
		if (predBoxes.size() == 0) {
			std::cout << " -> No hands found in the image!\n";
		}
		std::cout << " \nPixel accuracy for image " + std::to_string(i) + " --> " + std::to_string(accuracy) + "\n";
	}
}