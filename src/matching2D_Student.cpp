/**
 * matching2D_Student.cpp
 * 
 * This file contains the functions for the Mid-Term Project: 2D Feature Tracking
 * 
 * Author: Morsinaldo Medeiros
 * Date: 2024-01-03
 * 
 * Implementation reference: https://github.com/mohanadhammad/sfnd-2D-feature-tracking/blob/master/src/matching2D_Student.cpp
 * Github Copilot was used to help with the implementation
 * 
*/

#include <numeric>
#include "matching2D.hpp"

using namespace std;

extern int matchedPoints;
extern int kptsNumber;
extern float detectorTime;
extern float descriptorTime;
extern std::string detectorType;
extern std::string descriptorType;

/**
 * matchDescriptors
 * 
 * Detect keypoints in image using the traditional Harris detector
 * 
 * @param kPtsSource keypoints in the source image
 * @param kPtsRef keypoints in the reference image
 * @param descSource descriptors in the source image
 * @param descRef descriptors in the reference image
 * @param matches matches between the source and reference image
 * @param descriptorType descriptor type
 * @param matcherType matcher type
 * @param selectorType selector type
 *  
 * @return void
*/
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        int normType = cv::NORM_HAMMING;
        if (descriptorType.compare("DES_BINARY") == 0) {
            normType = cv::NORM_HAMMING;
        } else if (descriptorType.compare("DES_HOG") == 0) {
            normType = cv::NORM_L2;
        } else {
            throw std::invalid_argument("Invalid descriptor type: " + descriptorType);
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching";

    } else if (matcherType.compare("MAT_FLANN") == 0) {
        // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        } 
        
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";
    } else {
        throw std::invalid_argument("Invalid matcher type: " + matcherType);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) { 
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); 
    } else if (selectorType.compare("SEL_KNN") == 0){ 
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;

        // Finds the best match for each descriptor in desc1
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
            // if the best match has a distance ratio less than minDescDistRatio, then keep it
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }

        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

/**
 * descKeypoints
 * 
 * Use one of several types of state-of-art descriptors to uniquely identify keypoints
 * 
 * @param keypoints keypoints in the image
 * @param img image
 * @param descriptors descriptors in the image
 * @param descriptorType descriptor type
 * 
 * @return void
*/
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) {
        // BRIEF is a binary descriptor
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0) {
        // ORB is a binary descriptor
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0) {
        // FREAK is a binary descriptor
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0) {
        // AKAZE is a binary descriptor
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0) {
        // SIFT is a floating point descriptor
        extractor = cv::xfeatures2d::SIFT::create();
    } else {
        // invalid descriptor type
        throw std::invalid_argument("Invalid descriptor type: " + descriptorType);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    descriptorTime = 1000 * t / 1.0;
    cout << descriptorType << " descriptor extraction in " << descriptorTime << " ms" << endl;
}

/**
 * detKeypointsHarris
 * 
 * Detect keypoints in image using the traditional Harris detector
 * 
 * @param keypoints keypoints in the image
 * @param img image
 * @param bVis visualize results
 * 
 * @return void
*/
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // Detecting corners
    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); // convert to 8 bit

    // look for prominent corners and instantiate keypoints
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse) {
                // only store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > 0.0) {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap) {
                    // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    detectorTime = 1000 * t / 1.0;
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;

    // visualize results
    if (bVis) {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

/**
 * detKeypointsShiTomasi
 * 
 * Detect keypoints in image using the traditional Shi-Thomasi detector
 * 
 * @param keypoints keypoints in the image
 * @param img image
 * @param bVis visualize results
 * 
 * @return void
*/
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    detectorTime = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

/**
 * detKeypointsModern
 * 
 * Detect keypoints in image using the modern detectors
 * 
 * @param keypoints keypoints in the image
 * @param img image
 * @param detectorType detector type
 * @param bVis visualize results
 * 
 * @return void
*/
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {
    
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0) {
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;   // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    } else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
    } else if (detectorType.compare("ORB") == 0) {
        int nfeatures = 30000;
        detector = cv::ORB::create(nfeatures);
    } else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    } else if (detectorType.compare("SIFT") == 0) {
        int nfeatures = 10000;
        detector = cv::xfeatures2d::SIFT::create(nfeatures);
    } else {
        throw std::invalid_argument("Invalid detector type: " + detectorType);
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {
        string windowName = detectorType + " Detector Results";
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}