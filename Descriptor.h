#include "dirent.h"
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <cmath>
#include <string.h>
#include <math.h>

#define PI_CONST 3.14159265358979323846
#define CELL_SIDE_SIZE 8
#define BIN_NUMBER 9
#define NUMBER_OF_CELLS_IN_BLOCK 4
#define ROWS 128
#define COLS 64

/**
 * Structure for gradient feature
 */
typedef struct {
    int verticalGradient[ROWS][COLS];
    int horizontalGradient[ROWS][COLS];
    double magnitudes[ROWS][COLS];
    double angles[ROWS][COLS];
} GradientFeature;

/**
 * Descriptor class
 */
class Descriptor {
private:
    std::vector<cv::Mat> posList, fullNegList, negList, gradientList;
    std::vector<int> labels;
    std::string posDirectory = "positivesPath/";
    std::string negDirectory = "negativesPath/";
    std::string objDetFilename = "output.vec";
    std::string testDirectory = "testImagesPath/";
    cv::Size posImageSize;
    bool onlyTesting = !true;
    bool trainAndTest = true;

/**
* Function tests images from test directory with trained machine
*
* @param testDirectory test directory
* @param svm trained svm
*/
    void test(std::string testDirectory, cv::Ptr<cv::ml::SVM> svm) {
        std::vector<cv::Mat> testList;
        std::vector<std::string> imgNames = Descriptor::loadImages(testDirectory, testList);

        int listSize = testList.size();
        int wrongDetectionsPos = 0;
        int wrongDetectionsNeg = 0;

        if (testList.size() < 1) {
            std::cerr << "No tests loaded!" << std::endl;
            exit(1);
        }

        for (int i = 0; i < listSize; i++) {
            std::vector<cv::Mat> images, grad;
            images.push_back(testList[i]);
            computeHOGs(images, grad);

            cv::Mat testData;
            convertToMl(grad, testData);

            std::vector<float> responses;
            svm->predict(testData, responses);

            for (float number : responses) {
                std::cout << "Image " << imgNames[i] << ": " << number << std::endl;
                if (checkStart(imgNames[i], 0) && number < 0) {
                    wrongDetectionsPos++;
                }
                if (checkStart(imgNames[i], 1) && number > 0) {
                    wrongDetectionsNeg++;
                }
            }
        }

        std::cout << "Pictures: " << listSize << std::endl;
        std::cout << "Wrong detections in positives: " << wrongDetectionsPos << std::endl;
        std::cout << "Wrong detections in negatives: " << wrongDetectionsNeg << std::endl;
    }

/**
* Function loads images from directory
 *
 * @param directory path to directory
 * @param imageList image list
 *
 * @return image name list
*/
    std::vector<std::string> loadImages(std::string &directory, std::vector<cv::Mat> &imageList) {

        char *directoryPath = &directory[0u];
        std::vector<std::string> retList;
        for (cv::String item : readDirectory(directoryPath)) {
            retList.push_back(item);
            item = directory + item;

            cv::Mat image = cv::imread(item);

            if (image.data == 0) {
                std::cerr << "Image " << item << " cannot be loaded!" << std::endl;
                continue;
            }

            cv::Mat outputImage = adjustImage(image);
            imageList.push_back(outputImage);
        }
        return retList;

    }

/**
* Function computes HOG
 *
 * @param imageList image list
 * @param gradientList gradient list
*/
    void computeHOGs(std::vector<cv::Mat> &imageList, std::vector<cv::Mat> &gradientList) {
        for (int i = 0; i < imageList.size(); i++) {
            std::vector<float> descriptors = calculateImageHistogram(imageList[i]);
            gradientList.push_back(cv::Mat(descriptors).clone());
        }
    }

/**
* Function adjusts image
 *
 * @param image input image
 *
 * @return adjusted image
*/
    cv::Mat adjustImage(cv::Mat image) {
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        cv::Mat outputImage = resizeImage(image);
        return outputImage;
    }

/**
* Function converts samples to machine language
 *
 * @param trainSamples train samples
 * @param trainData train data
*/
    void convertToMl(const std::vector<cv::Mat> &trainSamples, cv::Mat &trainData) {
        //--Convert data
        const int rows = (int) trainSamples.size();
        const int cols = (int) std::max(trainSamples[0].cols, trainSamples[0].rows);

        cv::Mat tmp(1, cols, CV_32FC1);                    // used for transposition if needed
        trainData = cv::Mat(rows, cols, CV_32FC1);

        for (int i = 0; i < trainSamples.size(); ++i) {
            CV_Assert(trainSamples[i].cols == 1 || trainSamples[i].rows == 1);
            if (trainSamples[i].cols == 1) {
                transpose(trainSamples[i], tmp);
                tmp.copyTo(trainData.row((int) i));
            } else if (trainSamples[i].rows == 1) {
                trainSamples[i].copyTo(trainData.row((int) i));
            }
        }
    }

/**
 * Function resizes photo to 128x64
 *
 * @param image original image
 * @return resized image
 */
    cv::Mat resizeImage(cv::Mat image) {
        cv::Mat outputImage;
        cv::resize(image, outputImage, cv::Size(64, 128));
        return outputImage;
    }

/**
 * Function reads all files from directory
 *
 * @param string path to directory
 * @return vector of all files names
 */
    std::vector<std::string> readDirectory(char *string) {
        DIR *dir;
        struct dirent *directory;
        std::vector<std::string> nameVector;

        dir = opendir(string);
        if (dir) {
            while ((directory = readdir(dir)) != NULL) {
                if (checkEnd(directory->d_name)) {
                    nameVector.push_back(directory->d_name);
                }
            }

            closedir(dir);
        } else {
            throw std::runtime_error("Directory cannot be open!");
        }

        return nameVector;
    }

/**
 * Function checks if file extension is '.jpeg', '.jpg' or '.png'
 *
 * @param string file name
 *
 * @return true is file's extension is right, otherwise false
 */
    bool checkEnd(std::string string) {
        std::vector<std::string> ending = {".png", ".jpg", ".jpeg"};

        for (std::string end : ending) {
            if (string.length() >= end.length()) {
                if (string.compare(string.length() - end.length(), end.length(), end) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

/**
 * Function checks is image name starts with correct prefix depending on return value from svm.predict
 * @param string image name
 * @param 1 is return value is negative, 0 if value is positive
 *
 * @return <code>true</code> if image name starts with appropriate prefix, otherwise <code>false</code>
 */
    bool checkStart(std::string string, int number) {
        std::vector<std::string> starting = {"pos", "neg"};
        std::string start = starting[number];

        if (string.length() >= start.length()) {
            if (string.compare(0, start.length(), start) == 0) {
                return true;
            }
        }

        return false;
    }

/**
* Function that calculates gradient features (horizontal coordinate, vertical coordinate,
* horizontal gradient, vertical gradient, magnitude and angle in degrees) for given pixel
*
* @return GradientFeature structure for given pixel
*/
    GradientFeature *gradientFeatureCalculator(cv::Mat &matrix) {
        GradientFeature *gradFeature = new GradientFeature;

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                if (i == 0 || i == ROWS - 1 || j == 0 || j == COLS - 1) {
                    gradFeature->horizontalGradient[i][j] = 0;
                    gradFeature->verticalGradient[i][j] = 0;
                    gradFeature->magnitudes[i][j] = 0;
                    gradFeature->angles[i][j] = 0;
                } else {
                    gradFeature->horizontalGradient[i][j] =
                            (int) matrix.at<uchar>(i, j + 1) - (int) matrix.at<uchar>(i, j - 1);
                    gradFeature->verticalGradient[i][j] =
                            (int) matrix.at<uchar>(i + 1, j) - (int) matrix.at<uchar>(i - 1, j);
                    gradFeature->magnitudes[i][j] = sqrt(pow(gradFeature->horizontalGradient[i][j], 2)
                                                         + pow(gradFeature->verticalGradient[i][j], 2));
                    gradFeature->angles[i][j] = (
                            atan2(gradFeature->verticalGradient[i][j], gradFeature->horizontalGradient[i][j])
                            * (180.0 / PI_CONST));
                    if (gradFeature->angles[i][j] < 0) {
                        gradFeature->angles[i][j] += 180.0;
                    }
                    if (std::abs(gradFeature->angles[i][j]) >= 180) {
                        gradFeature->angles[i][j] = 0;
                    }
                }
            }
        }
        return gradFeature;
    }

/**
* Function that calculates values in histogram bins of an image(64x128 pixels)
* @param image
* @return vector of doubles that represent histogram bins of the image
*/
    std::vector<float> calculateImageHistogram(cv::Mat image) {
        GradientFeature *gf = gradientFeatureCalculator(image);
        std::vector<float> hogDescriptorVec;

        int l = 0;
        for (int i = 0; i < ROWS - CELL_SIDE_SIZE; i += CELL_SIDE_SIZE) {
            for (int j = 0; j < COLS - CELL_SIDE_SIZE; j += CELL_SIDE_SIZE) {
                double *block = calculateBlockHistogram(*gf, i, j);
                int blockBins = BIN_NUMBER * NUMBER_OF_CELLS_IN_BLOCK;
                for (int k = 0; k < blockBins; k++) {
                    hogDescriptorVec.push_back((float) block[k]);
                }
                delete[] block;
            }
        }
        return hogDescriptorVec;
    }

/**
* Function that calculates values in histogram bins of one block(16x16 pixels). Values are normalized.
* @param gf gradient feature structure of image
* @param rowStart number of the first row of the block
* @param colStart number of the first column of the block
* @return vector of doubles that represent histogram bins of the block
*/
    double *calculateBlockHistogram(GradientFeature &gf, int rowStart, int colStart) {
        double *blockHistogram = new double[BIN_NUMBER * NUMBER_OF_CELLS_IN_BLOCK];
        double magnitudeNormValue = 0;

        double *cellUL = calculateCellHistogram(gf, rowStart, colStart);
        double *cellUR = calculateCellHistogram(gf, rowStart, colStart + CELL_SIDE_SIZE);
        double *cellLL = calculateCellHistogram(gf, rowStart + CELL_SIDE_SIZE, colStart);
        double *cellLR = calculateCellHistogram(gf, rowStart + CELL_SIDE_SIZE, colStart + CELL_SIDE_SIZE);

        double *cells[4] = {cellUL, cellUR, cellLL, cellLR};

        for (int i = 0; i < NUMBER_OF_CELLS_IN_BLOCK; i++) {
            for (int j = 0; j < BIN_NUMBER; j++) {
                magnitudeNormValue += cells[i][j] * cells[i][j]; //block normalization
            }
        }

        magnitudeNormValue = sqrt(magnitudeNormValue);

        for (int i = 0; i < NUMBER_OF_CELLS_IN_BLOCK; i++) {
            for (int j = 0; j < BIN_NUMBER; j++) {
                if (magnitudeNormValue > 0.0) {
                    blockHistogram[i * BIN_NUMBER + j] = cells[i][j] / magnitudeNormValue; //block normalization
                }
            }
        }
        delete[] cellUL;
        delete[] cellUR;
        delete[] cellLL;
        delete[] cellLR;

        return blockHistogram;
    }

/**
* Function that calculates values in histogram bins of one cell(8x8 pixels)
* @param gf gradient feature structure of image
* @param rowStart number of the first row of the cell
* @param colStart number of the first column of the cell
* @return vector of doubles that represent histogram bins of the cell
*/
    double *calculateCellHistogram(GradientFeature &gf, int rowStart, int colStart) {
        double *cellHistogram = new double[BIN_NUMBER];

        for (int i = 0; i < 9; i++) {
            cellHistogram[i] = 0.;
        }

        for (int i = rowStart; i < rowStart + CELL_SIDE_SIZE; i++) {
            for (int j = colStart; j < colStart + CELL_SIDE_SIZE; j++) {
                int binNum = (int) ((gf.angles[i][j]) / 20);

                int binNumShared;
                double percentage;
                if (fmod(gf.angles[i][j], 20) > 10.0) {
                    if (binNum != 8) {
                        binNumShared = binNum + 1;
                    } else {
                        binNumShared = 0;
                    }
                    percentage = fmod(gf.angles[i][j], 20) / 20.;
                } else {
                    if (binNum != 0) {
                        binNumShared = binNum - 1;
                    } else {
                        binNumShared = 8;
                    }
                    percentage = (10. + fmod(gf.angles[i][j], 20)) / 20.;
                }

                cellHistogram[binNum] += percentage * gf.magnitudes[i][j];
                cellHistogram[binNumShared] += (1 - percentage) * gf.magnitudes[i][j];
            }
        }
        return cellHistogram;
    }

public:
    void start() {
        if (onlyTesting) {
            cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(objDetFilename);


            test(testDirectory, svm);

            return;
        }

        loadImages(posDirectory, posList);
        if (posList.size() < 1) {
            std::cerr << "No positives loaded!" << std::endl;
            exit(1);
        }
        posImageSize = posList[0].size();

        loadImages(negDirectory, negList);
        if (negList.size() < 1) {
            std::cerr << "No negatives loaded!" << std::endl;
            exit(1);
        }

        int numOfPositives, numOfNegatives;
        labels.clear();

        computeHOGs(posList, gradientList);
        numOfPositives = gradientList.size();
        labels.assign(numOfPositives, +1);

        computeHOGs(negList, gradientList);
        numOfNegatives = gradientList.size() - numOfPositives;
        labels.insert(labels.end(), numOfNegatives, -1);

        cv::Mat trainData;
        convertToMl(gradientList, trainData);

        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::LINEAR);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10000, 1e-6));
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, labels);
        svm->train(td);

        if (trainAndTest) {
            test(testDirectory, svm);
        }


        svm->save(objDetFilename);
    }
};
