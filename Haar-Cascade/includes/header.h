#ifndef HEADER_H
#define HEADER_H
#endif // !HEADER_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

void FromKeypointToBox(std::string pathDataset);

void createFilePositiveSamples(std::string pathDataset);

void createFileNegativeSamples(std::string pathDataset);

void ToGrayscale(std::string pathDataset);

void BndBoxConvert(std::string pathDataset);

cv::Mat CascadeDetection(std::string model, cv::Mat img);

std::vector<float> toFloatVec(std::vector<double> vec);

void StoreAndDisplay(cv::Mat img, std::string NameWindow, std::string NameImage);