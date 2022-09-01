#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>								// used for remove last n character 
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core_detect.hpp>
#include <fstream>								// for creating file
#include <math.h>								// for floor function

#include "../includes/header.h"
#include "../includes/rapidxml/rapidxml.hpp"	// For parsing the xml
#include "../includes/tinyxml2.h"				// For creating the xml files
#include <filesystem>



/*
* @brief It compute the Pixel Accuracy metrics
* @bgMask The ground truth mask
* @predMask The predicted mask
* @return It returns the pixel accuracy value
*/
double pixelAcc(cv::Mat gtMask, cv::Mat predMask) {

	int TP = 0;	// true positive
	int TN = 0; // true negative
	int FP = 0; // false positive
	int FN = 0; // false negative 

	for (int x = 0; x < gtMask.cols; x++) {
		for (int y = 0; y < gtMask.rows; y++) {
			if (gtMask.at<unsigned char>(y, x) == 255) {
				if (predMask.at<unsigned char>(y, x) == 255) {
					TP++;
				}
				else {
					FN++;
				}
			}
			else {
				if (predMask.at<unsigned char>(y, x) == 0) {
					TN++;
				}
				else {
					FP++;
				}
			}
		}
	}

	double pixelAccuracy = (double)(TP + TN) / (TP + TN + FP + FN);

	return pixelAccuracy;
}


/*
* @brief It compute the IoU over two boxes
* @box1 A bounding box
* @box2 A bounding box
* @return It returns the IoU value between box1 and box2
*/
double iou(cv::Rect box1, cv::Rect box2) {
	int x1_gt = box1.x;
	int y1_gt = box1.y;
	int x2_gt = x1_gt + box1.width;
	int y2_gt = y1_gt + box1.height;

	// area ground truth box
	int area_gt = box1.width * box1.height;

	// area pred box
	int area_p = box2.width * box2.height;

	// compute intersection area
	int x1_p = box2.x;
	int y1_p = box2.y;
	int x2_p = x1_p + box2.width;
	int y2_p = y1_p + box2.height;

	int x1I = std::max(x1_gt, x1_p);
	int y1I = std::max(y1_gt, y1_p);
	int x2I = std::min(x2_gt, x2_p);
	int y2I = std::min(y2_gt, y2_p);	

	// check if they overlap
	if (x2I < x1I) {
		return 0;
	}

	if (y2I < y1I) {
		return 0;
	}

	int widthI = x2I - x1I + 1;
	int heightI = y2I - y1I + 1;

	int interArea =widthI * heightI;
	double union_area = area_gt + area_p - interArea;

	// compute IoU
	return (interArea / union_area);
}



/*
* @brief It compute the IoU for all the predicted boxes
* @groundBoxes The lsit of grouning boxes
* @predBoxes The list of predicted boxes
* @return It returns the IoU associated to each predicted box
*/
std::vector<double> IoU(std::vector<cv::Rect> groundBoxes, std::vector<cv::Rect> predBoxes) {

	std::vector<std::vector<double>> boxes;
	std::vector< double >iouList;

	for (int i = 0; i < predBoxes.size(); i++) {

		double MaxIoU = 0;
		double IoU;

		for(int j = 0; j < groundBoxes.size(); j++) {
			
			// compute IoU
			IoU = iou(groundBoxes[j], predBoxes[i]);

			// associate to the predicted box only the max IoU computed with respect to the ground truth box
			if (IoU > MaxIoU) {
				MaxIoU = IoU;
			}
		}

		// for the i-predicted box, we associate the max IoU computed
		iouList.push_back(MaxIoU);
	}

	return iouList;
}


/**
* @brief Return the groundTruth bounding boxes from the xml files
* @param path: The path of the folder contatining the annotations in xml
* @return The vector contatining the boxes
*/
std::vector<cv::Rect> getTruthBoxes(std::string path) {

	std::vector<cv::Rect> groundTruth;

	// get gound truth
	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;

	// Read the xml file into a vector
	std::ifstream xmlFile(path);
	std::vector<char> buffer((std::istreambuf_iterator<char>(xmlFile)), std::istreambuf_iterator<char>());
	buffer.push_back('\0');

	// Parse the buffer
	document.parse<0>(&buffer[0]);

	// Find our root node
	root_node = document.first_node("annotation");

	// Iterate over the box object
	for (rapidxml::xml_node<>* obj = root_node->first_node("object"); obj; obj = obj->next_sibling())
	{
		// Interate over the boxes
		rapidxml::xml_node<>* box = obj->first_node("bndbox");

		// get the box info
		rapidxml::xml_node<>* xmin = box->first_node("xmin");
		rapidxml::xml_node<>* xmax = box->first_node("xmax");
		rapidxml::xml_node<>* ymin = box->first_node("ymin");
		rapidxml::xml_node<>* ymax = box->first_node("ymax");

		int Xmin = strtol(xmin->value(), NULL, 10);
		int Ymin = strtol(ymin->value(), NULL, 10);
		int Xmax = strtol(xmax->value(), NULL, 10);
		int Ymax = strtol(ymax->value(), NULL, 10);

		cv::Rect bndBox = cv::Rect(Xmin, Ymin, Xmax - Xmin, Ymax - Ymin);

		groundTruth.push_back(bndBox);
	}

	return groundTruth;
}


/**
* @brief Read boxes from a txt file in the format
* @param path: The path of the folder contatining the files
* @return The vector contatining the boxes
*/
std::vector<cv::Rect> getBoxesFromTxt(std::string path) {

	std::vector<cv::Rect> boxes;

	// open and read file
	std::ifstream inFile;
	inFile.open(path);
	
	// check opening
	if (!inFile) {
		std::cerr << "Unable to open file datafile.txt";
		exit(EXIT_FAILURE);   
	}

	// get boxes 
	std::string line;
	while (std::getline(inFile, line))
	{
		// parse line
		std::stringstream lineStream(line);
		std::string token;
		std::vector<int> points;

		while (lineStream >> token)
		{
			points.push_back(stoi(token));
		}

		int x1 = points[0];
		int y1 = points[1];
		int width = points[2];
		int height = points[3];

		boxes.push_back(cv::Rect(x1, y1, width, height));
	}
	// close file
	inFile.close();

	return boxes;
}


/**
* @brief Updates the boxes after an image has been resized
* @param imageSizeOld: The size of the image before resizing
* @param imageSizeNew: The size of the image after resizing
* @return The vector contatining the resized boxes
*/
std::vector<cv::Rect> rescaleB(std::vector<cv::Rect> boxes, cv::Size imageSizeOld, cv::Size imageSizeNew) {

	float xScale = (float)imageSizeNew.width / imageSizeOld.width;
	float yScale = (float)imageSizeNew.height / imageSizeOld.height;
	
	std::vector<cv::Rect> out;

	// compute the new box coordinates
	for (int i = 0; i < boxes.size(); i++) {
		int newXmin = floor(boxes[i].x * xScale);
		int newYmin = floor(boxes[i].y * yScale);
		int newWidth = floor(boxes[i].width * xScale);
		int newHeight = floor(boxes[i].height * yScale);
		
		out.push_back(cv::Rect(newXmin, newYmin, newWidth, newHeight));
	}

	return out;
}


/**
* @brief Build the bounding box of an object from points describing a contour
* @param pathDataset: The path of the folder contatining the annotations with the point in xml format
* @param trainFolderXml: The path of the folder where to store the new xmls
* @return It stores in the folder pathDataset a file xml for each xml file read. The new xml file reports (x,y, width, height) for each contour of the objects, the name of the image and its size.
*/
void FromKeypointToBox(std::string pathDataset, std::string trainFolderXml) {

	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;
	
	std::string path = pathDataset + "/*.xml";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> xmlList;
	cv::glob(path, xmlList);

	//Read each xml file
	for (int i = 0; i < xmlList.size(); i++)
	{
		// we create a xml file for each image
		tinyxml2::XMLDocument  xmlDoc;
		tinyxml2::XMLNode* root = xmlDoc.NewElement("annotation");
		xmlDoc.InsertFirstChild(root);

		// Read the xml file into a vector
		std::ifstream xmlFile(xmlList[i]);
		std::vector<char> buffer((std::istreambuf_iterator<char>(xmlFile)), std::istreambuf_iterator<char>());
		buffer.push_back('\0');

		// Parse the buffer
		document.parse<0>(&buffer[0]);

		// Find our root node
		root_node = document.first_node("annotation");

		// start parsing
		rapidxml::xml_node<>* image = root_node->first_node("filename");
		char* filename = image->value();

		//add filename to the new xml
		tinyxml2::XMLElement* imageName = xmlDoc.NewElement("filename");
		imageName->SetText(filename);
		root->InsertEndChild(imageName);

		// add image size to the new xml
		rapidxml::xml_node<>* image_size = root_node->first_node("imagesize");
		tinyxml2::XMLElement* imagesize = xmlDoc.NewElement("size");
		root->InsertEndChild(imagesize);

		// get the size
		rapidxml::xml_node<>* image_width = image_size->first_node("ncols");
		rapidxml::xml_node<>* image_height = image_size->first_node("nrows");
		tinyxml2::XMLElement* val;
		val = xmlDoc.NewElement("width");
		val->SetText(image_width->value());
		imagesize->InsertEndChild(val);

		val = xmlDoc.NewElement("height");
		val->SetText(image_height->value());
		imagesize->InsertEndChild(val);


		// Iterate over the hands object
		for (rapidxml::xml_node<>* hand = root_node->first_node("object"); hand; hand = hand->next_sibling())
		{	
			// add object instance to the new xml
			tinyxml2::XMLElement* obj = xmlDoc.NewElement("object");
			root->InsertEndChild(obj);

			std::vector<cv::Point> contour; // used for storing the contour points

			// Interate over the polygons
			rapidxml::xml_node<>* polygon = hand->first_node("polygon");
			for (rapidxml::xml_node<>* point = polygon->first_node("pt"); point; point = point->next_sibling())
			{
				// get the points
				rapidxml::xml_node<>* x = point->first_node("x");
				rapidxml::xml_node<>* y = point->first_node("y");

				// convert the point from char* -> int
				int x_ = strtol(x->value(), NULL, 10);
				int y_ = strtol(y->value(), NULL, 10);

				contour.push_back(cv::Point(x_, y_));
			}

			// we compute the box coordinates for each hand contour
			cv::Rect box = cv::boundingRect(contour);

			int Xmin = box.x;			// x-top-left point of the box 
			int Ymin = box.y;			// y-top-left point of the box
			int height = box.height;	//box heighht
			int width = box.width;		//box width

			rapidxml::xml_node<>* image_size = root_node->first_node("imagesize");
			rapidxml::xml_node<>* image_width = image_size->first_node("ncols");
			rapidxml::xml_node<>* image_height = image_size->first_node("nrows");

			int imageWidth = strtol(image_width->value(), NULL, 10);
			int imageHeight = strtol(image_height->value(), NULL, 10);

			// check if box goes outside border of the image and updates
			if (Ymin + height > imageHeight) {
				height = imageHeight - Ymin;
			}

			// check if box goes outside border of the image and updates
			if (Xmin + width > imageWidth) {
				width = imageWidth - Xmin;
			}
			
			//add box to the new xml
			tinyxml2::XMLElement* bndbox = xmlDoc.NewElement("bndbox");
			obj->InsertEndChild(bndbox);

			tinyxml2::XMLElement* val;
			val = xmlDoc.NewElement("xmin");
			val->SetText(Xmin);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymin");
			val->SetText(Ymin);
			bndbox->InsertEndChild(val);

			int Xmax = Xmin + width;	// x-bottom-right point of the box
			int Ymax = Ymin + height;	// y-bottom-right point of the x
			val = xmlDoc.NewElement("xmax");
			val->SetText(Xmax);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymax");
			val->SetText(Ymax);
			bndbox->InsertEndChild(val);

		}

		std::string filenameToString = filename;
		std::string xmlFileName = filenameToString.erase(filenameToString.length() - 4); // we remove the '.jpg' from the filename

		// store xml file
		std::string fileName = trainFolderXml + "/" + xmlFileName + ".xml";
		tinyxml2::XMLError eResult = xmlDoc.SaveFile(fileName.c_str());
		if (eResult != 0) {
			std::cout << "File " + fileName + " could not be stored!\n";
		}

	}
}


/**
* @brief Copy images from one folder to another 
* @param path1: path folder containing the images to copy
* @param path2: path of the folder where to copy the images
*/
void copyImg(std::string path1, std::string path2) {

	std::string imagesPath = path1 + "/*.jpg";

	// Extracting all the images stored in the directory 'Dataset'
	std::vector<cv::String> imageList = loadImgs(path1);

	for (int i = 0; i < imageList.size(); i++)
	{
		// get the file name of each image
		std::string base_filename = imageList[i].substr(imageList[i].find_last_of("/\\") + 1);

		cv::Mat img = cv::imread(imageList[i]);
		cv::imwrite(path2 + "/" + base_filename, img);
	}
}


/**
* @brief Copy xml files from one folder to another
* @param path1: path folder containing the xmls to copy
* @param path2: path of the folder where to copy the xmls
*/
void copyXml(std::string path1, std::string path2) {

	std::string imagesPath = path1 + "/*.xml";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> xmlList;
	cv::glob(imagesPath, xmlList);

	for (int i = 0; i < xmlList.size(); i++)
	{
		// get the file name of each xml
		std::string base_filename = xmlList[i].substr(xmlList[i].find_last_of("/\\") + 1);

		std::ifstream src(xmlList[i], std::ios::binary);
		std::ofstream dest(path2 + "/" + base_filename + ".xml", std::ios::binary);
		dest << src.rdbuf();
	}
}


/**
* @brief Load xml files
* @param pathFiles: path folder containing the xmls to load
* @return The string vector containing the list of the path of the xml files
*/
std::vector<cv::String> loadXmls(std::string pathFiles) {
	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;

	std::string path = pathFiles + "/*.xml";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> xmlList;
	cv::glob(path, xmlList);

	if (xmlList.size() == 0) {
		std::cout << "Xml files not present in the folder!";
		exit(EXIT_FAILURE);
	}
	sort(xmlList.begin(), xmlList.end());

	return xmlList;
}


/**
* @brief Load txt files
* @param pathFiles: path folder containing the txt files to load
* @return The string vector containing the list of the path of the txt files
*/
std::vector<cv::String> loadTxt(std::string pathFiles) {
	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;

	std::string path = pathFiles + "/*.txt";

	// Extracting all the txt stored in the directory
	std::vector<cv::String> fileTxt;
	cv::glob(path, fileTxt);

	if (fileTxt.size() == 0) {
		std::cout << "Xml files not present in the folder!";
		exit(EXIT_FAILURE);
	}
	sort(fileTxt.begin(), fileTxt.end());

	return fileTxt;
}


/**
* @brief Load images files jpg and png
* @param pathFiles: path folder containing the images to load
* @return The Mat vector containing the loaded images
*/
std::vector<cv::Mat> loadImages(std::string pathFiles) {

	std::string jpg = pathFiles + "/*.jpg";
	std::string png = pathFiles + "/*.png";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> imageListJpg;
	cv::glob(jpg, imageListJpg);

	std::vector<cv::String> imageListPng;
	cv::glob(png, imageListPng);

	std::vector<cv::String> imageList;
	imageList = imageListJpg;

	for (int i = 0; i < imageListPng.size(); i++) {
		imageList.push_back(imageListPng[i]);
	}

	std::vector<cv::Mat> imgList;
	for (int i = 0; i < imageList.size(); i++) {
		cv::Mat img = cv::imread(imageList[i]);
		imgList.push_back(img);
	}

	if (imgList.size() == 0) {
		std::cout << "Images not present in the folder!";
		exit(EXIT_FAILURE);
	}

	sort(imageList.begin(), imageList.end());

	return imgList;
}


/**
* @brief Load the path of the images files jpg and png
* @param pathFiles: path folder containing the images to load
* @return The string vector containing the path of the images
*/
std::vector<cv::String> loadImgs(std::string pathFiles) {

	std::string jpg = pathFiles + "/*.jpg";
	std::string png = pathFiles + "/*.png";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> imageListJpg;
	cv::glob(jpg, imageListJpg);

	std::vector<cv::String> imageListPng;
	cv::glob(png, imageListPng);

	std::vector<cv::String> imageList;
	imageList = imageListJpg;

	for (int i = 0; i < imageListPng.size(); i++) {
		imageList.push_back(imageListPng[i]);
	}

	if (imageList.size() == 0) {
		std::cout << "Images not present in the folder!";
		exit(EXIT_FAILURE);
	}

	return imageList;
}


/**
* @brief Get the list of all the horizontal boxes from a file annotation
* @param xmlFile: path of the xml file annotation
* @return The rect vector containing all the horizontal boxes
*/
std::vector<cv::Rect>  getHorizontalBoxes(std::string xmlFile) {

	std::vector<cv::Rect> boxeH;

	// get gound truth
	std::vector<cv::Rect> boxes = getTruthBoxes(xmlFile);

	for (int j = 0; j < boxes.size(); j++) {

		// horizontal box
		if (boxes[j].width > boxes[j].height) {
			boxeH.push_back(boxes[j]);
		}
	}

	return boxeH;
}


/**
* @brief Get the list of all the vertical boxes from a file annotation
* @param xmlFile: path of the xml file annotation
* @return The rect vector containing all the vertical boxes
*/
std::vector<cv::Rect>  getVerticalBoxes(std::string xmlFile) {

	std::vector<cv::Rect> boxeV;

	// get gound truth
	std::vector<cv::Rect> boxes = getTruthBoxes(xmlFile);

	for (int j = 0; j < boxes.size(); j++) {

		// horizontal box
		if (boxes[j].width < boxes[j].height) {
			boxeV.push_back(boxes[j]);
		}
	}

	return boxeV;
}


/**
* @brief Get the list of all the horizontal boxes from multiple file annotation
* @param xmlList: the string vector containing the path of the xml annotation
* @return The rect vector containing all the horizontal boxes from the annotations provided
*/
std::vector<cv::Rect>  getAllHorizontalBoxes(std::vector<cv::String> xmlList) {

	std::vector<cv::Rect> boxesH;

	//Read each xml file
	for (int i = 0; i < xmlList.size(); i++)
	{
		// get horizontal boxes for each xml file
		std::vector<cv::Rect> boxes = getHorizontalBoxes(xmlList[i]);

		for (int j = 0; j < boxes.size(); j++) {
			boxesH.push_back(boxes[j]);
		}
	}

	return boxesH;

}

/**
* @brief Get the list of all the vertical boxes from multiple file annotation
* @param xmlList: the string vector containing the path of the xml annotation
* @return The rect vector containing all the vertical boxes from the annotations provided
*/
std::vector<cv::Rect>  getAllVerticalBoxes(std::vector<cv::String> xmlList) {

	std::vector<cv::Rect> boxesV;

	//Read each xml file
	for (int i = 0; i < xmlList.size(); i++)
	{
		// get vertical boxes for each xml file
		std::vector<cv::Rect> boxes = getVerticalBoxes(xmlList[i]);

		for (int j = 0; j < boxes.size(); j++) {
			boxesV.push_back(boxes[j]);
		}
	}

	return boxesV;

}


/**
* @brief Get the average size from a list of boxes
* @param box: the list of boxes
* @return The average size of the boxes in the list
*/
cv::Size  getAverageBoxSize(std::vector<cv::Rect> box) {

	// compute average width and height for vertical boxes
	int aveWidth = 0;
	int aveHeight = 0;
	float width = 0;
	float height = 0;

	for (int i = 0; i < box.size(); i++) {
		width = width + box[i].width;
		height = height + box[i].height;
	}

	aveWidth = floor(width / box.size());
	aveHeight = floor(height / box.size());

	return cv::Size(aveWidth, aveHeight);
}


/**
* @brief Visualize the boxes in the image from its annotation
* @param pathImg: the path of the image
* @param xmlPath: the path of the annotation
* @return Show the boxes on the image
*/
void visualizeBoxes(std::string pathImg, std::string xmlPath) {

	std::string base_filename;

	std::vector<cv::String> imageList = loadImgs(pathImg);
	std::vector<cv::String> xmlList = loadXmls(xmlPath);

	for (int i = 0; i < imageList.size(); i++) {
		cv::Mat img = cv::imread(imageList[i]);
		std::vector<cv::Rect> box1 = getHorizontalBoxes(xmlList[i]);
		std::vector<cv::Rect> box2 = getVerticalBoxes(xmlList[i]);

		for (int j = 0; j < box1.size(); j++) {
			cv::rectangle(img, box1[j], cv::Scalar(0, 255, 0));
		}

		for (int j = 0; j < box2.size(); j++) {
			cv::rectangle(img, box2[j], cv::Scalar(0, 255, 0));
		}

		base_filename = imageList[i].substr(imageList[i].find_last_of("/\\") + 1);
		cv::imshow(base_filename, img);
		cv::waitKey(0);
		cv::destroyWindow(base_filename);
	}
}


/**
* @brief Rescale the image to a specific width and set height so that to keep the original aspect ratio. It updates also the xml annotation
* @param pathImage: The path of the folder contatining the image
* @param pathXml: The path of the annotation
* @param pathStore: The path where to store the new image
* @param targetW: The new width of the new image
* #return It stores in pathStore the rescaled image.
*/
void rescaleKeepRatio(std::string pathImage, std::string pathXml, std::string pathStore,  int targetW) {

	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;

	std::vector<std::string> imageList = loadImgs(pathImage);
	std::vector<std::string> xmlList = loadXmls(pathXml);

	// reorder list
	sort(imageList.begin(), imageList.end());
	sort(xmlList.begin(), xmlList.end());

	for (int i = 0; i < imageList.size(); i++)
	{
		cv::Mat tmp = cv::imread(imageList[i]);
		cv::Size imgSize = tmp.size();

		// calculate the ratio of the width and construct the dimensions
		float r = float(targetW) / imgSize.width;
		int height = floor((imgSize.height * r));

		// rescaled
		cv::Mat output;
		cv::resize(tmp, output, cv::Size(targetW, height), cv::INTER_CUBIC);

		// get name
		std::string base_filename = imageList[i].substr(imageList[i].find_last_of("/\\") + 1);
		std::string imgName = base_filename.erase(base_filename.length() - 4); // we remove the '.jpg' from the filename

		// store
		cv::imwrite(pathStore + "/" + imgName + ".jpg", output);

		// UPDATES XML
		// we create the xml file
		tinyxml2::XMLDocument  xmlDoc;
		tinyxml2::XMLNode* root = xmlDoc.NewElement("annotation");
		xmlDoc.InsertFirstChild(root);

		// Read the xml file into a vector
		std::ifstream xmlFile(xmlList[i]);
		std::vector<char> buffer((std::istreambuf_iterator<char>(xmlFile)), std::istreambuf_iterator<char>());
		buffer.push_back('\0');

		// Parse the buffer
		document.parse<0>(&buffer[0]);

		// Find our root node
		root_node = document.first_node("annotation");

		// start parsing
		rapidxml::xml_node<>* image = root_node->first_node("filename");
		char* filename = image->value();

		//add filename to the new xml
		tinyxml2::XMLElement* imageName = xmlDoc.NewElement("filename");
		imageName->SetText(filename);
		root->InsertEndChild(imageName);

		// add image size to the new xml
		tinyxml2::XMLElement* imagesize = xmlDoc.NewElement("size");
		root->InsertEndChild(imagesize);

		tinyxml2::XMLElement* val;
		val = xmlDoc.NewElement("width");
		val->SetText(targetW);
		imagesize->InsertEndChild(val);

		val = xmlDoc.NewElement("height");
		val->SetText(height);
		imagesize->InsertEndChild(val);

		float xScale = (float)targetW / imgSize.width;
		float yScale = (float)height / imgSize.height;

		// Iterate over the hands object
		for (rapidxml::xml_node<>* hand = root_node->first_node("object"); hand; hand = hand->next_sibling())
		{
			// Interate over the boxes
			rapidxml::xml_node<>* box = hand->first_node("bndbox");

			// get the box info
			rapidxml::xml_node<>* xmin = box->first_node("xmin");
			rapidxml::xml_node<>* xmax = box->first_node("xmax");
			rapidxml::xml_node<>* ymin = box->first_node("ymin");
			rapidxml::xml_node<>* ymax = box->first_node("ymax");

			int Xmin = strtol(xmin->value(), NULL, 10);
			int Ymin = strtol(ymin->value(), NULL, 10);
			int Xmax = strtol(xmax->value(), NULL, 10);
			int Ymax = strtol(ymax->value(), NULL, 10);

			// compute the new box coordinates
			int newXmin = floor(Xmin * xScale);
			int newYmin = floor(Ymin * yScale);
			int newXmax = floor(Xmax * xScale);
			int newYmax = floor(Ymax * yScale);

			// update xml file
			// add only those box coordinates whose new width and height are not 0
			int newW = newXmax - newXmin;
			int newH = newYmax - newYmin;

			if (newW != 0 && newH != 0) {
				// add object instance to the new xml
				tinyxml2::XMLElement* obj = xmlDoc.NewElement("object");
				root->InsertEndChild(obj);

				tinyxml2::XMLElement* bndbox = xmlDoc.NewElement("bndbox");
				obj->InsertEndChild(bndbox);

				tinyxml2::XMLElement* val;
				val = xmlDoc.NewElement("xmin");
				val->SetText(newXmin);
				bndbox->InsertEndChild(val);

				val = xmlDoc.NewElement("ymin");
				val->SetText(newYmin);
				bndbox->InsertEndChild(val);

				val = xmlDoc.NewElement("xmax");
				val->SetText(newXmax);
				bndbox->InsertEndChild(val);

				val = xmlDoc.NewElement("ymax");
				val->SetText(newYmax);
				bndbox->InsertEndChild(val);
			}

		}
		// get name
		base_filename = xmlList[i].substr(xmlList[i].find_last_of("/\\") + 1);
		std::string xmlFileName = base_filename.erase(base_filename.length() - 4); // we remove the '.xml' from the filename

		//store
		std::string fileName = pathStore + "/" + xmlFileName + ".xml";

		tinyxml2::XMLError eResult = xmlDoc.SaveFile(fileName.c_str());
		if (eResult != 0) {
			std::cout << "File " + fileName + " could not be stored!\n";
		}
	}
}


/**
* @brief Rescale the image to a specific width and set height so that to keep the original aspect ratio
* @param img: The path of the folder contatining the image
* @param targetW: The new width of the new image
* #return It returns the rescaled image
*/
cv::Mat rescaleImg(cv::Mat img, int targetW) {

	cv::Size imgSize = img.size();

	// calculate the ratio of the width and construct the dimensions
	float r = float(targetW) / imgSize.width;
	int height = floor((imgSize.height * r));

	// rescaled
	cv::Mat output;
	cv::resize(img, output, cv::Size(targetW, height), cv::INTER_CUBIC);

	return output;
}

/*
* @brief Utlity function to convert a double vector to a float vector
* @param vec : The double vector to convert
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