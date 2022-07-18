#ifndef HEADER_H
#define HEADER_H
#endif // !HEADER_H

void FromKeypointToBox(std::string pathDataset);

void createFilePositiveSamples(std::string pathDataset);

void createFileNegativeSamples(std::string pathDataset);

void ToGrayscale(std::string pathDataset);

void BndBoxConvert(std::string pathDataset);
