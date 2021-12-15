#ifndef ELSED_UTILS_H_
#define ELSED_UTILS_H_

#include <ostream>
#include <opencv2/opencv.hpp>

#define UPM_ABS(x) ((x) >= 0 ? (x) : -(x))

#define UPM_EDGE_VERTICAL   0
#define UPM_EDGE_HORIZONTAL 255

#define UPM_ED_NO_EDGE_PIXEL 0
#define UPM_ED_EDGE_PIXEL    255
#define UPM_ED_ANCHOR_PIXEL  204
#define UPM_ED_SEGMENT_INLIER_PX 153
#define UPM_ED_SEGMENT_OUTLIER_PX 102
#define UPM_ED_JUNTION_PX 152

#define UPM_LEFT  1
#define UPM_RIGHT 2
#define UPM_UP    3
#define UPM_DOWN  4

#define UPM_SKIP_EDGE_PT 2
#define UPM_MAX_OUTLIERS_TH 3

namespace upm {
// Line segment in format [x0, y0, x1, y1] where the endpoints are (x0, y0) and (x1, y1)
typedef cv::Vec4f Segment;
typedef std::vector<Segment> Segments;

/**
 * Data structure containing the gradient information of an image
 */
struct LineDetectionExtraInfo {
  cv::Mat dxImg;   // Store the dxImg (horizontal gradient)
  cv::Mat dyImg;   // Store the dyImg (vertical gradient)
  cv::Mat gImgWO;  // Store the gradient image without threshold
  cv::Mat gImg;    // Store the gradient image
  cv::Mat dirImg;  // Store the direction image

  // image sizes
  unsigned int imageWidth;
  unsigned int imageHeight;
};
// Pointer to the struct
typedef std::shared_ptr<LineDetectionExtraInfo> LineDetectionExtraInfoPtr;

/**
 * Simple data structure representing a pixel. Unlike cv::Point2i this class is
 * trivial (trivially copyable and has a trivial default constructor).
 */
struct Pixel {
  int x;
  int y;

  Pixel() = default;
  Pixel(int x, int y) : x(x), y(y) {}
  explicit Pixel(const cv::Point &p) : x(p.x), y(p.y) {}

  inline bool operator==(const Pixel &rhs) const {
    return x == rhs.x && y == rhs.y;
  }
  inline bool operator!=(const Pixel &rhs) const {
    return !(rhs == *this);
  }

  inline bool operator<(const Pixel &px2) const {
    // Compare the pixels by its squared L1 distance to the origin
    return (x + y) < (px2.x + px2.y);
  }
};

// Define the edge types
typedef std::vector<Pixel> ImageEdge;
typedef std::vector<ImageEdge> ImageEdges;

/**
 * This struct represents a segment weighted by its importance in the image.
 */
struct SalientSegment {
  Segment segment;
  double salience;

  SalientSegment() = default;
  SalientSegment(const Segment &segment, double salience) : segment(segment), salience(salience) {}

  inline bool operator<(const SalientSegment &rhs) const {
    if (salience == rhs.salience) {
      float dx1 = segment[0] - segment[2];
      float dx2 = rhs.segment[0] - rhs.segment[2];
      float dy1 = segment[1] - segment[3];
      float dy2 = rhs.segment[1] - rhs.segment[3];
      return std::sqrt(dx1 * dx1 + dy1 * dy1) > std::sqrt(dx2 * dx2 + dy2 * dy2);
    } else {
      return salience > rhs.salience;
    }
  }
};

typedef std::vector<SalientSegment> SalientSegments;

/**
 * @brief Calculate the angular distance between two angles valueA and valueB.
 * Taking into account that 0 degrees is equivalent to 360
 * @param valueA
 * @param valueB
 * @param mod The number of elements in the circle (2 * PI to radians or 360 for degrees)
 * @return
 */
inline double
circularDist(double valueA, double valueB, double mod = 360) {
  double a, b; // a is the small number and b the great number
  if (valueA < valueB) {
    a = valueA;
    b = valueB;
  } else {
    a = valueB;
    b = valueA;
  }

  double dist_clockwise, dist_no_clockwise;
  dist_clockwise = b - a;
  dist_no_clockwise = a + (mod - b);
  return std::min(dist_clockwise, dist_no_clockwise);
}

/**
 * @brief Gets the projection of a point into a line.
 * @param l The general line ecuation ax + by +c = 0 in format (a, b, c)
 * @param p The point which we want to calculate its projection over the line
 * @return The point p' which is the projection of p over the line.
 */
inline cv::Point2f
getProjectionPtn(const cv::Vec3f &l, const cv::Point2f &p) {
  const cv::Vec3f homoP(p.x, p.y, 1);
  if (l.dot(homoP) == 0) {
    // If the point is over the line return this same point
    return p;
  }
  // Since the direction of l is (-l.b, l.a), the rotated 90 degrees vector will be: (l.a, l.b)
  // The direction vector of the perpendicular rect we want to calc.
  const cv::Vec2f v2(l[0], l[1]);
  // Rect with direction v2 passing by point p
  const cv::Vec3f r2(v2[1], -v2[0], v2[0] * p.y - v2[1] * p.x);
  // return the intersection between the two lines in cartesian coordinates
  const cv::Vec3f p2 = l.cross(r2);
  return cv::Point2f(p2[0] / p2[2], p2[1] / p2[2]);
}

/**
 * @brief Calculates the length of a line segment
 * @param s The input segment
 * @return The length of the segment
 */
inline float
segLength(const Segment &s) {
  // The optimal way to do that, is to compute first the differences
  const float dx = s[0] - s[2];
  const float dy = s[1] - s[3];
  // And after that, do the square root, avoiding the use of double
  return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Calculates the angle of a line segment
 * @param s The input segment
 * @return The angle that the segment forms with the X-axis in radians.
 * Range (pi/2, -pi/2].
 */
inline float
segAngle(const Segment &s) {
  if (s[2] > s[0])
    return std::atan2(s[3] - s[1], s[2] - s[0]);
  else
    return std::atan2(s[1] - s[3], s[0] - s[2]);
}


/**
 * Returns the line pixels using the Bresenham Algorithm:
 * https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
 * @param x0 The X coordinate of the first endpoint
 * @param y0 The Y coordinate of the first endpoint
 * @param x1 The X coordinate of the last endpoint
 * @param y1 The Y coordinate of the last endpoint
 * @return
 */
static std::vector<Pixel> bresenham(int x0, int y0, int x1, int y1) {
  int dx, dy, p, x, y, xIncrement, yIncrement;
  std::vector<Pixel> pixels;
  dx = x1 - x0;
  dy = y1 - y0;

  // Determine the line direction
  xIncrement = dx < 0 ? -1 : +1;
  yIncrement = dy < 0 ? -1 : +1;

  x = x0;
  y = y0;
  dx = UPM_ABS(dx);
  dy = UPM_ABS(dy);
  // pixels.reserve(std::max(dx, dy));

  if (dx >= dy) {
    // Horizontal like line
    p = 2 * dy - dx;
    while (x != x1) {
      pixels.emplace_back(x, y);
      if (p >= 0) {
        y += yIncrement;
        p += 2 * dy - 2 * dx;
      } else {
        p += 2 * dy;
      }
      // Increment the axis in which we are moving
      x += xIncrement;
    }  // End of while
  } else {
    // Vertical like line
    p = 2 * dx - dy;
    while (y != y1) {
      pixels.emplace_back(x, y);
      if (p >= 0) {
        x += xIncrement;
        p += +2 * dx - 2 * dy;
      } else {
        p += 2 * dx;
      }
      // Increment the axis in which we are moving
      y += yIncrement;
    }  // End of while
  }
  pixels.emplace_back(x1, y1);
  return pixels;
}
}  // namespace upm

#endif  // ELSED_UTILS_H_
