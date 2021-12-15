#include <iostream>
#include <opencv2/opencv.hpp>
#include "ELSED.h"

inline void
drawSegments(cv::Mat img,
             upm::Segments segs,
             const cv::Scalar &color,
             int thickness = 1,
             int lineType = cv::LINE_AA,
             int shift = 0) {
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
}

int main() {
  std::cout << "******************************************************" << std::endl;
  std::cout << "******************* ELSED main demo ******************" << std::endl;
  std::cout << "******************************************************" << std::endl;

  // Using default parameters (long segments)
  cv::Mat img = cv::imread("../images/P1020829.jpg");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  upm::ELSED elsed;
  upm::Segments segs = elsed.detect(img);
  std::cout << "ELSED detected: " << segs.size() << " (large) segments" << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imshow("ELSED long", img);
  cv::waitKey();

  // Not using jumps (short segments)
  img = cv::imread("../images/P1020829.jpg");
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  upm::ELSEDParams params;
  params.listJunctionSizes = {};
  upm::ELSED elsed_short(params);
  segs = elsed_short.detect(img);
  std::cout << "ELSED detected: " << segs.size() << " (short) segments" << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imshow("ELSED short", img);
  cv::waitKey();

  return 0;
}