/**
 * @file camFusion_Student.cpp
 * 
 * @brief Functions for 3D object tracking
 * 
 * Author: Morsinaldo Medeiros
 * Date 2024-01-05
 * 
 * Implementation reference: https://github.com/mohanadhammad/sfnd-3d-object-tracking/blob/master/src/camFusion_Student.cpp
 * Github copilot was used to help with the implementation
*/
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

/**
 * Create groups of Lidar points whose projection into the camera falls into the same bounding box
 * 
 * @param boundingBoxes bounding boxes
 * @param lidarPoints Lidar points
 * @param shrinkFactor shrink factor
 * @param P_rect_xx rectified camera intrinsic matrix
 * @param R_rect_xx rectified camera extrinsic matrix
 * @param RT camera extrinsic matrix
 * 
 * @return void
*/
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/**
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*
* @param boundingBoxes bounding boxes
* @param worldSize world size
* @param imageSize image size
* @param bWait wait for key press
*
* @return void
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

/**
 * Associate a given bounding box with the keypoints it contains
 * 
 * @param boundingBox bounding box
 * @param keypoints keypoints
 * @param kptMatches keypoint matches
 * @param kptsPrev keypoints in previous frame
 * @param kptsCurr keypoints in current frame
 * 
 * @return void
*/
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double sum_distance = 0;
    std::vector<cv::DMatch> matches;

    // Find the mean distance between all matched keypoints
    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it) {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);

        if(boundingBox.roi.contains(kpCurr.pt)) {
            matches.push_back(*it);
            sum_distance += cv::norm(kpCurr.pt - kpPrev.pt);
        }
    }

    // Find the threshold distance
    double distMean = sum_distance / matches.size();
    double threshold = distMean * 0.7;

    // Find the matches that are within the threshold distance to filter out outliers
    for(auto it = matches.begin(); it != matches.end(); ++it) {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);

        if(cv::norm(kpCurr.pt - kpPrev.pt) < threshold) {
            boundingBox.kptMatches.push_back(*it);
        }
    }

    // Print the number of keypoints within the bounding box
    std::cout << "Number of keypoints within the bounding box: " << boundingBox.kptMatches.size() << std::endl;
}

/**
 * @brief Compute time-to-collision (TTC) based on keypoint correspondences in successive images
 * 
 * @param kptsPrev keypoints in previous frame
 * @param kptsCurr keypoints in current frame
 * @param kptMatches keypoint matches between previous and current frame
 * @param frameRate frame rate of the current camera
 * @param TTC time-to-collision (TTC) of the camera
 * @param visImg visualization image
 * 
 * @return void
*/
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> ratio_distances; // stores the distance ratios for all keypoints between curr. and prev. frame

    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1){
        
        const cv::KeyPoint &prev_keypoint1 = kptsPrev[it1->queryIdx];
        const cv::KeyPoint &curr_keypoint1 = kptsCurr[it1->trainIdx];

        for(auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2){

            double dist_min = 100.0; // min. required distance
            
            const cv::KeyPoint &prev_keypoint2 = kptsPrev[it2->queryIdx];
            const cv::KeyPoint &curr_keypoint2 = kptsCurr[it2->trainIdx];
            
            // compute distances and distance ratios
            double prev_distance = cv::norm(prev_keypoint1.pt - prev_keypoint2.pt);
            double curr_distance = cv::norm(curr_keypoint1.pt - curr_keypoint2.pt);

            if (curr_distance > std::numeric_limits<double>::epsilon() && prev_distance >= dist_min){
                // if the distance ratio is smaller than distThreshold, then the distance between the keypoints is valid
                ratio_distances.push_back(curr_distance / prev_distance);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (ratio_distances.size() == 0){
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    std::sort(ratio_distances.begin(), ratio_distances.end());
    long median_index = floor(ratio_distances.size() / 2.0);
    bool even = ratio_distances.size() % 2 == 0;
    double median_distance_ratio;

    if (even){
        median_distance_ratio = (ratio_distances[median_index - 1] + ratio_distances[median_index]) / 2.0;
    }
    else{
        median_distance_ratio = ratio_distances[median_index];
    }

    TTC = (-1.0 / frameRate) / (1 - median_distance_ratio);

    // print TTC
    std::cout << "TTC Camera: " << TTC << " s" << std::endl;
}

/**
 * @brief Compute time-to-collision (TTC) based on Lidar measurements
 * 
 * @param lidarPointsPrev Lidar points in previous frame
 * @param lidarPointsCurr Lidar points in current frame
 * @param frameRate frame rate of the current Lidar
 * @param TTC time-to-collision (TTC) of the Lidar
 * 
 * @return void
*/
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double lane_width = 4.0; // Assuming lane width is 4 meters
    std::vector<double> x_values_prev, x_values_curr;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it){
        // Filter out points that are not in the ego lane
        if(abs(it->y) <= lane_width / 2.0){
            x_values_prev.push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it){
        // Filter out points that are not in the ego lane
        if(abs(it->y) <= lane_width / 2.0){
            x_values_curr.push_back(it->x);
        }
    }

    // Sort the vectors to find the median
    std::sort(x_values_prev.begin(), x_values_prev.end());
    std::sort(x_values_curr.begin(), x_values_curr.end());

    // Compute the median
    double x_median_prev = x_values_prev[x_values_prev.size() / 2];
    double x_median_curr = x_values_curr[x_values_curr.size() / 2];

    // compute TTC from both measurements
    TTC = x_median_curr * (1.0 / frameRate) / (x_median_prev - x_median_curr);

    // print TTC
    std::cout << "TTC Lidar: " << TTC << " s" << std::endl;
}

/**
 * @brief Match bounding boxes between current and previous frame
 * 
 * @param matches vector of matches between current and previous frame
 * @param bbBestMatches map of best matches between bounding boxes of current and previous frame
 * @param prevFrame previous frame
 * @param currFrame current frame
 * 
 * @return void
*/
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame){

    // use the matched keypoints to find the bounding boxes that contain them
    // loop over all matches

    cv::Mat match_table = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);

    for (const auto &match : matches){
        const cv::KeyPoint &prev_keypoint = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &curr_keypoint = currFrame.keypoints[match.trainIdx];

        // find the bounding boxes that contain the matched keypoints
        for (const auto &prev_bbox : prevFrame.boundingBoxes){
            if (prev_bbox.roi.contains(prev_keypoint.pt)){
                for (const auto &curr_bbox : currFrame.boundingBoxes){
                    if (curr_bbox.roi.contains(curr_keypoint.pt)){
                        match_table.at<int>(prev_bbox.boxID, curr_bbox.boxID) += 1;
                    }
                }
            }
        }
    }
    
    // find the best matches
    for (int i = 0; i < match_table.rows; i++){
        
        int max_count = 0;
        int max_id = -1;

        for (int j = 0; j < match_table.cols; j++){
            if (match_table.at<int>(i, j) > max_count && match_table.at<int>(i, j) > 0){
                max_count = match_table.at<int>(i, j);
                max_id = j;
            }
        }

        if (max_id != -1){
            bbBestMatches.emplace(i, max_id);
        }
    }
}
