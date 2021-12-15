#ifndef ELSED_FULLSEGMENTINFO_H_
#define ELSED_FULLSEGMENTINFO_H_

#include <ostream>
#include <memory>
#include "Utils.h"

namespace upm {

/**
 * This class represents a line segment and its associated information such as:
 * - List of supporting pixels
 * - Line equation
 * - Endpoints
 *
 * The implementation is highly optimized to add or subtract new pixels and recompute all parameters.
 * Internally,
 */
class FullSegmentInfo {
 public:
  // Creates the FullSegmentInfo object but do not initialize internal structures
  FullSegmentInfo(const std::vector<Pixel> &pts);
  // Creates the FullSegmentInfo object and initializes internal structures
  FullSegmentInfo(const std::vector<Pixel> &pts, int startIdx);
  // Initialize the internal data structure from scratch
  void init(const std::vector<Pixel> &pts, int startIdx);
  // Checks if the point (x, y) is an inlier given the current line equation and a distance th.
  inline bool isInlier(int x, int y, double lineFitErrThreshold) {
    const double pointToLineDis = equation[0] * x + equation[1] * y + equation[2];
    return UPM_ABS(pointToLineDis) < lineFitErrThreshold;
  }

  // Adds a new pixel (x, y) to the segment and update its parameters.
  // pixelIndexInEdge is the index of the pixel in the edge structure.
  // isPixelAtTheEnd True if the pixel is at the end of the segment or false if it is at the start
  void addPixel(int x, int y, int pixelIndexInEdge, bool isPixelAtTheEnd = true);

  // Finishes the detection of segment.
  void finish();

  void skipPositions();

  inline double getFitError() const {
    double dist, fitError = 0;
    for (int i = firstPxIndex; i <= lastPxIndex; i++) {
      dist = equation[0] * (*pixels)[i].x + equation[1] * (*pixels)[i].y + equation[2];
      fitError += dist * dist;
    }
    return fitError / N;
  }

  void removeLastPx(bool removeFromTheEnd = true);

  inline void reset() {
    sum_x_i = 0, sum_y_i = 0, sum_x_i_y_i = 0, sum_x_i_2 = 0, N = 0;
    firstPxIndex = -1;
    lastPxIndex = -1;
    arePixelsSorted = true;
    firstEndpointExtended = false;
    secondEndpointExtended = false;
  }

  ///////////////////// Getters and setters /////////////////////

  inline bool horizontal() const { return isHorizontal; }

  inline int getNumOfPixels() const { return N; }

  inline const Segment &getEndpoints() const { return endpoints; }

  ImageEdge getPixels() const;

  //Returns the Pixel in the first extreme of the segment
  inline const Pixel &getFirstPixel() const { return firstPx; }

  //Returns the Pixel in the first extreme of the segment
  inline const Pixel &getLastPixel() const { return lastPx; }

  // Returns a pointer to the first pixel of the segment
  inline const Pixel *begin() const { return &((*pixels)[firstPxIndex]); }

  // Returns a pointer to the position after the last pixel of the segment
  inline const Pixel *end() const { return (&((*pixels)[lastPxIndex])) + 1; }

  inline const cv::Vec3d &getLineEquation() const { return equation; }

  inline bool hasSecondSideElements() const { return !arePixelsSorted; }

  // True if the segment has been already extended in the direction of the first endpoint
  bool firstEndpointExtended = false;
  // Idem for the second one
  bool secondEndpointExtended = false;

 private:
  // Fits the line segment from scratch
  void leastSquareLineFit(const std::vector<Pixel> &pts, int startIndex = 0);
  // Adds a new pixel to the internal structures
  void leastSquaresLineFitNewPoint(int x, int y);
  // Subtracts a pixel from the internal structures
  void subtractPointFromModel(const Pixel &p);
  // Computes the line equation given the internal parameters (very efficient)
  void calculateLineEq();
  // Computes the segment endpoints given the internal parameters (very efficient)
  void calcSegmentEndpoints();

  int64_t sum_x_i = 0, sum_y_i = 0, sum_x_i_y_i = 0, sum_x_i_2 = 0;
  uint32_t N = 0;
  int dx, dy;
  // Tru if the line is mainly horizontal. When the line is closer to vertical,
  // we invert the least squares and use y as independent coordinate and x as dependent coord.
  bool isHorizontal;
  // Pre-computed line segment endpoints
  Segment endpoints;
  // Line equation in format ax + by + c = 0
  cv::Vec3d equation;
  // Important pixels that are part of the segment
  Pixel firstPx, prevFirstPx, lastPx, prevLastPx;
  // Indices of some pixels in the chain
  int firstPxIndex = -1, lastPxIndex = -1;
  // Relevance of the segment
  float salience = -1;
  // Pointer to the chain of edge pixels
  const ImageEdge *pixels;
  // Indicates whenever the list of pixels from firstPxIndex to lastPxIndex is sorted
  bool arePixelsSorted = true;
};

}

#endif //ELSED_FULLSEGMENTINFO_H_
