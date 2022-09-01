#ifndef HEADER_H
#define HEADER_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


/* ----------------------------- Utilities -----------------------------------------*/
void FromKeypointToBox(std::string pathDataset, std::string trainFolderXml);

void ToGrayscale(std::vector<cv::String> imageList);

void copyImg(std::string path1, std::string path2);

void copyXml(std::string path1, std::string path2);

std::vector<cv::Rect> getTruthBoxes(std::string path);

cv::Size getImageSize(std::string xml);

void createSamples(std::string imageList, std::string xmlList, std::string path2, int targetW, int targetH, int padding);

void createNegativeHogSamplesFromPositive(std::string imageList, std::string xmlList, std::string path2, int targetW, int targetH);

cv::Size  getAverageBoxSize(std::vector<cv::Rect> box);

std::vector<cv::Rect>  getVerticalBoxes(std::string xmlFile);

std::vector<cv::Rect>  getHorizontalBoxes(std::string xmlFile);

std::vector<cv::Rect>  getAllHorizontalBoxes(std::vector<cv::String> xmlList);

std::vector<cv::Rect>  getAllVerticalBoxes(std::vector<cv::String> xmlList);

std::vector<cv::String> loadXmls(std::string path);

std::vector<cv::String> loadImgs(std::string pathFiles);

std::vector<cv::Mat> loadImages(std::string pathFiles);

void visualizeBoxes(std::string pathImg, std::string xmlPath);

std::vector<cv::Size> getWindowsSize(std::string positiveAnnotations);

std::vector<float> toFloatVec(std::vector<double> vec);

void augmentData(std::string negativeCropped);

void rescaleKeepRatio(std::string pathImage, std::string pathXml, std::string negStore, int targetW);

std::vector<cv::String> loadTxt(std::string pathFiles);

std::vector<cv::Rect> getBoxesFromTxt(std::string path);

cv::Mat rescaleImg(cv::Mat img, int targetW);

std::vector<cv::Rect> rescaleB(std::vector<cv::Rect> boxes, cv::Size imageSizeOld, cv::Size imageSizeNew);

void dataSetUp(std::string image_dataset, std::string image_annotations);

void createDataset(std::string image_dataset, std::string image_annotations, std::string negPath);


/* ----------------------   DETECTION AND INSTANCE SEGMENTATION -----------------------------*/
double iou(cv::Rect box1, cv::Rect box2);

std::vector<double> IoU(std::vector<cv::Rect> groundBoxes, std::vector<cv::Rect> predBoxes);

double pixelAcc(cv::Mat bgMask, cv::Mat predMask);

std::vector<cv::Mat> computeHog(cv::Size wsize, std::vector<cv::String> img_lst);

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);

std::vector< float > get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm);

void SVMdetector(std::string SVM_ModeFile, cv::Mat img, cv::Size winsize, std::vector< cv::Rect >& finalBox, std::vector<float>& finalScores);

void createHogDetector(std::string positiveImages, std::string negativeDataCropped, std::string pathModel, cv::Size winsize);

void trainSVM(std::vector<cv::Mat> gradient_lst, std::vector< int > labels, std::string path, cv::Size winsize);

void test_trained_detector(std::string SVM_ModeFile, cv::Mat img, cv::Size winsize, std::vector< cv::Rect >& finalBox, std::vector<float>& finalScores, float scoreThreshold, float nmsThreshold);

std::vector<cv::Rect> detectBox(cv::Mat img, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2);

cv::Mat InstanceSegContour(cv::Mat seg, std::vector<cv::Rect> prediction);

cv::Mat instaSegGrabcut(cv::Mat seg, std::vector<cv::Rect> prediction);

cv::Mat instaTH(cv::Mat img, std::vector<cv::Rect> prediction);

bool skinDetector(cv::Mat bnd);

void buildSamples(std::string pathP, std::string pathStore, std::string negStore, cv::Size winSize, int padding);

void detectHands(std::vector< cv::Mat > images, std::vector< std::string > annotations, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2);

void instanceSegmentation(std::vector< cv::Mat > images, std::vector< cv::Mat > gtMasks, std::string SVM1, std::string SVM2, cv::Size winsize1, cv::Size winsize2);


#endif // !HEADER_H