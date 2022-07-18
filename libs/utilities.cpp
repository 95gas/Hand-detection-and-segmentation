#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include "../includes/header.h"
#include <iostream>
#include <string>  // used for remove last n character 
#include <fstream>
#include <vector>
#include <opencv2/highgui.hpp>
#include "../includes/rapidxml/rapidxml.hpp" // For parsing the xml
#include "../includes/tinyxml2.h"		     // For creating the xml files
#include <fstream>							// for creating file


/**
* @brief Convert the (xmin, xmax, ymin, ymax) to (xmin, xmax, width, height)
* @param pathDataset: The path of the folder contatining the annotations in xml format
* @return It stores in the folder pathDataset/boxAnnotations/ a file xml for each xml file read. The new xml file reports (x,y, width, height) for each object.
*/
void BndBoxConvert(std::string pathDataset) {

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

		// Iterate over the hands object
		for (rapidxml::xml_node<>* hand = root_node->first_node("object"); hand; hand = hand->next_sibling())
		{
			// add object instance to the new xml
			tinyxml2::XMLElement* obj = xmlDoc.NewElement("object");
			root->InsertEndChild(obj);

			// Interate over the boxes
			rapidxml::xml_node<>* box = hand->first_node("bndbox");

			// get the box info
			rapidxml::xml_node<>* xmin = box->first_node("xmin");
			rapidxml::xml_node<>* xmax = box->first_node("xmax");
			rapidxml::xml_node<>* ymin = box->first_node("ymin");
			rapidxml::xml_node<>* ymax = box->first_node("ymax");

			char* coor_x = xmin->value();
			char* coor_y = ymin->value();

			// compute height box
			int y_max = strtol(ymax->value(), NULL, 10);
			int y_min = strtol(ymin->value(), NULL, 10);
			int height =  y_max - y_min;

			// compute width box
			int x_max = strtol(xmax->value(), NULL, 10);
			int x_min = strtol(xmin->value(), NULL, 10);
			int width = x_max - x_min;

			//add box to the new xml
			tinyxml2::XMLElement* bndbox = xmlDoc.NewElement("bndbox");
			obj->InsertEndChild(bndbox);

			tinyxml2::XMLElement* val;
			val = xmlDoc.NewElement("xmin");
			val->SetText(coor_x);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymin");
			val->SetText(coor_y);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("xmax");
			val->SetText(width);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymax");
			val->SetText(height);
			bndbox->InsertEndChild(val);

		}

		std::string filenameToString = filename;
		std::string xmlFileName = filenameToString.erase(filenameToString.length() - 4); // we remove the '.jpg' from the filename

		std::string fileName = pathDataset + "/boxAnnotations/" + xmlFileName + ".xml";

		tinyxml2::XMLError eResult = xmlDoc.SaveFile(fileName.c_str());
		if (eResult != 0) {
			std::cout << "File " + fileName + " could not be stored!\n";
		}

	}
}

/**
* @brief Build the bounding box of an object using points describing a contour
* @param pathDataset: The path of the folder contatining the annotations with the point in xml format
* @return It stores in the folder pathDataset/boxAnnotations/ a file xml for each xml file read. The new xml file reports (x,y, width, height) for each contour of the objects.
*/
void FromKeypointToBox(std::string pathDataset) {

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

			// we compute the box coordinates
			cv::Rect box = cv::boundingRect(contour);

			int coor_x = box.x;
			int coor_y = box.y;
			int height = box.height;
			int width = box.width;

			// check if box goes outside border of the image
			rapidxml::xml_node<>* image_size = root_node->first_node("imagesize");
			rapidxml::xml_node<>* image_width = image_size->first_node("ncols");
			rapidxml::xml_node<>* image_height = image_size->first_node("nrows");

			int imageWidth = strtol(image_width->value(), NULL, 10);
			int imageHeight = strtol(image_height->value(), NULL, 10);

			if (coor_y + height > imageHeight) {
				height = imageHeight - coor_y;
			}

			if (coor_x + width > imageWidth) {
				width = imageWidth - coor_x;
			}

			//add box to the new xml
			tinyxml2::XMLElement* bndbox = xmlDoc.NewElement("bndbox");
			obj->InsertEndChild(bndbox);

			tinyxml2::XMLElement* val;
			val = xmlDoc.NewElement("xmin");
			val->SetText(coor_x);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymin");
			val->SetText(coor_y);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("xmax");
			val->SetText(width);
			bndbox->InsertEndChild(val);

			val = xmlDoc.NewElement("ymax");
			val->SetText(height);
			bndbox->InsertEndChild(val);

		}

		std::string filenameToString = filename;
		std::string xmlFileName = filenameToString.erase(filenameToString.length() - 4); // we remove the '.jpg' from the filename

		std::string fileName = pathDataset + "/boxAnnotations/" + xmlFileName + ".xml";
		
		tinyxml2::XMLError eResult = xmlDoc.SaveFile(fileName.c_str());
		if (eResult != 0) {
			std::cout << "File " + fileName + " could not be stored!\n";
		}

	}
}



/**
* @brief Create the file listing the positive samples. The images are listing according to the format required by the opencv_createsamples application.
* @param pathDataset: The path of the folder contatining the positive samples annotations
* #return It stores a file .dat listing each image with the number of box presented on it and its (x,y,width, height) information for each box. 
*/
void createFilePositiveSamples(std::string pathDataset) {

	// Create and open a dat file
	std::ofstream MyFile(pathDataset + "/positive.dat");

	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root_node;

	std::string path = pathDataset + "/*.xml";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> xmlList;
	cv::glob(path, xmlList);

	//Read each xml file
	for (int i = 0; i < xmlList.size(); i++)
	{
		std::string line = "positive/";

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

		// update line to write in the file
		line = line + filename + " ";

		int count = 0; // count the hands found

		std::string boxCoordinates = "";

		// Iterate over the box object
		for (rapidxml::xml_node<>* obj = root_node->first_node("object"); obj; obj = obj->next_sibling())
		{
			count++;

			// Interate over the boxes
			rapidxml::xml_node<>* box = obj->first_node("bndbox");

			// get the box info
			rapidxml::xml_node<>* xmin = box->first_node("xmin");
			rapidxml::xml_node<>* xmax = box->first_node("xmax");
			rapidxml::xml_node<>* ymin = box->first_node("ymin");
			rapidxml::xml_node<>* ymax = box->first_node("ymax");

			boxCoordinates = boxCoordinates + ' ' + xmin->value() + ' ' + ymin->value() + ' ' + xmax->value() + ' ' + ymax->value();

		}

		line = line + std::to_string(count) + boxCoordinates + "\n";

		// add lines to file
		MyFile << line;
	}

	// Close the file
	MyFile.close();
}


/**
* @brief Create the file listing the negative samples. The images are listing according to the format required by the opencv_createsamples application.
* @param pathDataset: The path of the folder contatining the negative samples. The images should be in .jpg format.
* #return It stores a file .dat listing each image contained in the pathDataset folder
*/
void createFileNegativeSamples(std::string pathDataset) {
	// Create and open a dat file
	std::ofstream MyFile(pathDataset + "/negative.dat");

	std::string path = pathDataset + "/*.jpg";

	// Extracting all the xml stored in the directory 'Dataset'
	std::vector<cv::String> imageList;
	cv::glob(path, imageList);

	for (int i = 0; i < imageList.size(); i++)
	{
		std::string line = "negative/";

		// get the file name of each image
		std::string base_filename = imageList[i].substr(imageList[i].find_last_of("/\\") + 1);

		// update line to write in the file
		line = line + base_filename + "\n";

		// add lines to file
		MyFile << line;
	}

	// Close the file
	MyFile.close();
}

// opencv_createsamples.exe -info 

/**
* @brief Convert all the images in a folder to grayscale.
* @param pathDataset: The path of the folder contatining the images in jpg extension
* #return It replace each image in the pathDataset folder with its grayscale version.
*/
void ToGrayscale(std::string pathDataset) {

	// Extracting all the xml stored in the directory 'Dataset'
	std::string path = pathDataset + "/*.jpg";

	std::vector<cv::String> imageList;
	cv::glob(path, imageList);

	for (int i = 0; i < imageList.size(); i++)
	{
		cv::Mat tmp = cv::imread(imageList[i]);

		// convert to grayscale only RGB images
		if (tmp.channels() == 3) {
			cv::Mat gray;
			cvtColor(tmp, gray, cv::COLOR_BGR2GRAY);

			// remove RGB image
			remove(imageList[i].c_str());

			// save grayscale image
			cv::imwrite(imageList[i], gray);
		}

	}
}