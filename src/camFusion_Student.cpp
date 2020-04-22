
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the
// same bounding box
void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints,
                         float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT)
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
    pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);  // pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

    vector<vector<BoundingBox>::iterator> enclosingBoxes;  // pointers to all
                                                           // bounding boxes
                                                           // which enclose the
                                                           // current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
    {
      // shrink current bounding box slightly to avoid having too many outlier
      // points around the edges
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

    }  // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1)
    {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  }  // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
  {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
    {
      // world coordinates
      float xw = (*it2).x;  // world position in m with x facing forward from sensor
      float yw = (*it2).y;  // world position in m with y facing left from sensor
      xwmin = xwmin < xw ? xwmin : xw;
      ywmin = ywmin < yw ? ywmin : yw;
      ywmax = ywmax > yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top < y ? top : y;
      left = left < x ? left : x;
      bottom = bottom > y ? bottom : y;
      right = right > x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0;  // gap between distance markers
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

  if (bWait)
  {
    cv::waitKey(0);  // wait for key to be pressed
  }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev,
                              std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches)
{
  std::vector<cv::DMatch> matches_in_roi;
  auto distance_mean = 0.0f;
  for (auto& match : kptMatches)
  {
    auto trainPt = kptsCurr[match.trainIdx].pt;
    auto queryPt = kptsPrev[match.queryIdx].pt;

    if (boundingBox.roi.contains(trainPt) || boundingBox.roi.contains(queryPt))
    {  // TODO Check
      matches_in_roi.push_back(match);
      distance_mean += match.distance;
    }
  }
  distance_mean /= kptMatches.size();
  for (auto& match : matches_in_roi)
  {
    if (std::abs(match.distance - distance_mean) / distance_mean < 5)
    {
      boundingBox.kptMatches.push_back(match);
    }
  }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, cv::Mat* visImg)
{
  std::vector<double> distanceRatios;
  for (auto& match : kptMatches)
  {
    // get current keypoint and its matched partner in the prev. frame
    auto kpOuterCurr = kptsCurr.at(match.trainIdx);
    auto kpOuterPrev = kptsPrev.at(match.queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
    {
      auto minDist = 100.0;  // min. required distance
      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
      if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
      {  // avoid division by zero

        double distRatio = distCurr / distPrev;
        distanceRatios.push_back(distRatio);
      }
    }  // eof inner loop over all matched kpts
  }    // eof outer loop over all matched kpts
  if (distanceRatios.size() == 0)
  {
    TTC = NAN;
    return;
  }

  std::sort(distanceRatios.begin(), distanceRatios.end());
  // compute camera-based TTC from distance ratios
  double meanDistRatio = std::accumulate(distanceRatios.begin(), distanceRatios.end(), 0.0) / distanceRatios.size();

  long medIndex = floor(distanceRatios.size() / 2.0);
  double medDistRatio = distanceRatios.size() % 2 == 0 ?
                            (distanceRatios[medIndex - 1] + distanceRatios[medIndex]) / 2.0 :
                            distanceRatios[medIndex];

  double dT = 1 / frameRate;
  TTC = -dT / (1 - medDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev, std::vector<LidarPoint>& lidarPointsCurr,
                     double frameRate, double& TTC)
{
  auto prevMinX = 100000.0;
  auto currMinX = 100000.0;
  constexpr float reflectivityThreshold = 0.4;

  std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(),
            [](const LidarPoint& lhs, const LidarPoint& rhs) { return lhs.x < rhs.x; });
  std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(),
            [&](const LidarPoint& lhs, const LidarPoint& rhs) { return lhs.x < rhs.x; });
  int medianIxPrev = lidarPointsPrev.size() / 2;
  int medianIxCurr = lidarPointsPrev.size() / 2;

  prevMinX = lidarPointsPrev[medianIxPrev].x;
  currMinX = lidarPointsCurr[medianIxCurr].x;

  auto deltaX = prevMinX - currMinX;
  auto dt = 1. / frameRate;
  auto velocity = deltaX / dt;

  TTC = currMinX / velocity;
}

void matchBoundingBoxes(std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches, DataFrame& prevFrame,
                        DataFrame& currFrame)
{
  std::multimap<int, int> potentialBoxMatches;
  for (auto& match : matches)
  {
    auto trainPt = currFrame.keypoints[match.trainIdx];
    auto queryPt = prevFrame.keypoints[match.queryIdx];

    auto currBBs = currFrame.boundingBoxes;
    auto prevBBs = prevFrame.boundingBoxes;
    for (auto& prevbb : prevBBs)
    {
      if (prevbb.roi.contains(queryPt.pt))
      {
        for (auto& currbb : currBBs)
        {
          if (currbb.roi.contains(trainPt.pt))
          {
            potentialBoxMatches.insert(std::pair<int, int>(prevbb.boxID, currbb.boxID));
          }
        }
      }
    }
  }
  // Count potential matches
  std::map<int, std::map<int, int>> prevBBToCurrBBcounter;  // prev -> current, count
  for (auto & [ prevbb, currbb ] : potentialBoxMatches)
  {
    ++prevBBToCurrBBcounter[prevbb][currbb];
  }
  for (auto & [ prevbb, bbCountPair ] : prevBBToCurrBBcounter)
  {
    std::map<int, int, std::greater<int>> countToBB;  // Reverse the map, so that the counts are keys. And higher counts
                                                      // occur first
    for (auto & [ bb, count ] : bbCountPair)
    {
      countToBB[count] = bb;
    }
    bbBestMatches[prevbb] = countToBB.begin()->second;
  }
}
