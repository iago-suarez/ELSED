#include <vector>
#include "FullSegmentInfo.h"

namespace upm {

FullSegmentInfo::FullSegmentInfo(const std::vector<Pixel> &pts) : pixels(&pts) {}

FullSegmentInfo::FullSegmentInfo(const std::vector<Pixel> &pts, int startIdx) : pixels(&pts) {
  init(pts, startIdx);
}

void FullSegmentInfo::init(const std::vector<Pixel> &pts, int startIdx) {
  assert(pts.size() > startIdx);
  assert(pts.size() >= 2);

  dx = pts.back().x - pts[startIdx].x;
  dy = pts.back().y - pts[startIdx].y;
  isHorizontal = UPM_ABS(dx) >= UPM_ABS(dy);

  // Fit the line
  leastSquareLineFit(pts, startIdx);

  firstPx = pts[startIdx];
  lastPx = pts.back();
  prevFirstPx = pts[startIdx + 1];
  prevLastPx = pts[pts.size() - 2];
  firstPxIndex = startIdx;
  lastPxIndex = pts.size() - 1;

  arePixelsSorted = true;
  firstEndpointExtended = false;
  secondEndpointExtended = false;
}

void FullSegmentInfo::skipPositions() {
  static_assert(UPM_SKIP_EDGE_PT == 2, "Error this code is optimized for UPM_SKIP_EDGE_PT = 2");
  assert(pixels->size() == lastPxIndex + 1 + UPM_SKIP_EDGE_PT);

  // Remove the last UPM_SKIP_EDGE_PT of the model (here UPM_SKIP_EDGE_PT == 2)
  subtractPointFromModel(firstPx);
  subtractPointFromModel(prevFirstPx);

  // Update the indices of the pixels we are using in the vector of pixels. Move all the indices
  firstPxIndex += UPM_SKIP_EDGE_PT;
  lastPxIndex += UPM_SKIP_EDGE_PT;
  firstPx = (*pixels)[firstPxIndex];
  lastPx = pixels->back();
  prevFirstPx = (*pixels)[firstPxIndex + 1];
  prevLastPx = (*pixels)[lastPxIndex - 1];

  dx = lastPx.x - firstPx.x;
  dy = lastPx.y - firstPx.y;
  bool tmp = UPM_ABS(dx) >= UPM_ABS(dy);
  if (tmp != isHorizontal) {
    isHorizontal = tmp;
    // Change of orientation
    leastSquareLineFit(*pixels, firstPxIndex);
  } else {
    leastSquaresLineFitNewPoint(prevLastPx.x, prevLastPx.y);
    leastSquaresLineFitNewPoint(lastPx.x, lastPx.y);
  }
}

void FullSegmentInfo::addPixel(int x, int y, int pixelIndexInEdge, bool isPixelAtTheEnd) {
  leastSquaresLineFitNewPoint(x, y);
  assert(!std::isnan(equation[0]) && !std::isnan(equation[1]));

  if (isPixelAtTheEnd) {
    prevLastPx = lastPx;
    lastPx.x = x;
    lastPx.y = y;
  } else {
    prevFirstPx = firstPx;
    firstPx.x = x;
    firstPx.y = y;
  }

  // The list of pixels is not sorted but should be contiguous in memory.
  // In consequence the last pixel index is always the last added
  lastPxIndex = pixelIndexInEdge;
  arePixelsSorted = arePixelsSorted && isPixelAtTheEnd;
}

void FullSegmentInfo::finish() {
  // Calculate the endpoints of the segment
  calcSegmentEndpoints();
  assert(!std::isnan(endpoints[0]));
}

inline void FullSegmentInfo::calcSegmentEndpoints() {
  // At last, compute the line endpoints and store them. We project the first
  // and last pixels in the pixelChain onto the best fit line to get the line endpoints.
  // xp= (w2^2 * x0 -w1*w2 * y0 -w3 * w1)
  // yp= (w1^2 * y0 -w1*w2 * x0 -w3 * w2)
  float a1 = equation[1] * equation[1];
  float a2 = equation[0] * equation[0];
  float a3 = equation[0] * equation[1];
  float a4 = equation[2] * equation[0];
  float a5 = equation[2] * equation[1];
  // First pixel
  const Pixel &firstPx = getFirstPixel();
  int Px = firstPx.x;
  int Py = firstPx.y;
  endpoints[0] = a1 * Px - a3 * Py - a4;  // x
  endpoints[1] = a2 * Py - a3 * Px - a5;  // y
  // Last pixel
  const Pixel &secondPx = getLastPixel();
  Px = secondPx.x;
  Py = secondPx.y;
  endpoints[2] = a1 * Px - a3 * Py - a4;  // x
  endpoints[3] = a2 * Py - a3 * Px - a5;  // y
}

ImageEdge FullSegmentInfo::getPixels() const {
  assert(firstPxIndex < lastPxIndex);
  ImageEdge result;
  result.reserve(getNumOfPixels());
  // Copy the elements to the destination vector
  for (const Pixel &px: *this)
    result.push_back(px);
  return result;
}

void FullSegmentInfo::removeLastPx(bool removeFromTheEnd) {
  // Subtract the pixel from the model parameters
  subtractPointFromModel(removeFromTheEnd ? lastPx : firstPx);
  // Update the endpoints information
  if (removeFromTheEnd) {
    assert(prevLastPx.x != -1);
    lastPx = prevLastPx;
    prevLastPx.x = -1;
    prevLastPx.y = -1;
  } else {
    assert(prevFirstPx.x != -1);
    firstPx = prevFirstPx;
    prevFirstPx.x = -1;
    prevFirstPx.y = -1;
  }
  lastPxIndex--;
}

inline void FullSegmentInfo::leastSquareLineFit(const std::vector<Pixel> &pts, int startIdx) {
  int i, indpCoord, depCoord;
  sum_x_i = 0, sum_y_i = 0, sum_x_i_y_i = 0, sum_x_i_2 = 0;
  N = pts.size() - startIdx;
  if (isHorizontal) {
    for (i = startIdx; i < pts.size(); i++) {
      indpCoord = pts[i].x;
      depCoord = pts[i].y;
      sum_x_i += indpCoord;
      sum_y_i += depCoord;
      sum_x_i_2 += indpCoord * indpCoord;
      sum_x_i_y_i += indpCoord * depCoord;
    }
  } else {
    for (i = startIdx; i < pts.size(); i++) {
      indpCoord = pts[i].y;
      depCoord = pts[i].x;
      sum_x_i += indpCoord;
      sum_y_i += depCoord;
      sum_x_i_2 += indpCoord * indpCoord;
      sum_x_i_y_i += indpCoord * depCoord;
    }
  }

  calculateLineEq();
}

inline void FullSegmentInfo::leastSquaresLineFitNewPoint(int x, int y) {
  const int indpCoord = isHorizontal ? x : y;
  const int depCoord = isHorizontal ? y : x;
  N++;
  sum_x_i += indpCoord;
  sum_y_i += depCoord;
  sum_x_i_2 += indpCoord * indpCoord;
  sum_x_i_y_i += indpCoord * depCoord;

  calculateLineEq();
}

inline void FullSegmentInfo::subtractPointFromModel(const Pixel &p) {
  const int independentCord = isHorizontal ? p.x : p.y;
  const int dependentCord = isHorizontal ? p.y : p.x;
  N--;
  sum_x_i -= independentCord;
  sum_y_i -= dependentCord;
  sum_x_i_2 -= independentCord * independentCord;
  sum_x_i_y_i -= independentCord * dependentCord;

  calculateLineEq();
}

inline void FullSegmentInfo::calculateLineEq() {
  // Line equation is calculated as:
  // 	\[
  //	ax + by + c = 0 \left\{
  //	\begin{array}{ll}
  //	a = N \cdot \sum (x_i y_i) - \sum x_i \cdot \sum y_i\\
  //	b = \left( \sum x_i \right )^2 - N \cdot \sum x_i^2\\
  //	c = \sum y_i \cdot \sum x_i^2 - \sum x_i \cdot \sum x_i y_i
  //	\end{array}
  //	\right.
  //	\]
  // Compute the line equation coefficients
  equation[2] = sum_y_i * sum_x_i_2 - sum_x_i * sum_x_i_y_i;
  if (isHorizontal) {
    equation[0] = N * sum_x_i_y_i - sum_x_i * sum_y_i;
    equation[1] = sum_x_i * sum_x_i - N * sum_x_i_2;
  } else {
    equation[1] = N * sum_x_i_y_i - sum_x_i * sum_y_i;
    equation[0] = sum_x_i * sum_x_i - N * sum_x_i_2;
  }

  // Normalize the line equation
  equation *= (1 / std::sqrt(equation[0] * equation[0] + equation[1] * equation[1]));

  // Fix the line equation
  if (dx * -equation[1] + dy * equation[0] < 0) {
    equation[0] = -equation[0];
    equation[1] = -equation[1];
    equation[2] = -equation[2];
  }
}

}
