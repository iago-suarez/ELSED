#ifndef ELSED_ELSED_H_
#define ELSED_ELSED_H_

#include <ostream>

#include "FullSegmentInfo.h"
#include "EdgeDrawer.h"

namespace upm {

struct ELSEDParams {
  // Gaussian kernel size
  int ksize = 5;
  // Sigma of the gaussian kernel
  float sigma = 1;
  // The threshold of pixel gradient magnitude.
  // Only those pixels whose gradient magnitude are larger than
  // this threshold will be taken as possible edge points.
  float gradientThreshold = 30;
  // If the pixel's gradient value is bigger than both of its neighbors by a
  // certain anchorThreshold, the pixel is marked to be an anchor.
  uint8_t anchorThreshold = 8;
  // Anchor testing can be performed at different scan intervals, i.e.,
  // every row/column, every second row/column
  unsigned int scanIntervals = 2;

  // Minimum line segment length
  int minLineLen = 15;
  // Threshold used to check if a list of edge points for a line segment
  double lineFitErrThreshold = 0.2;
  // Threshold used to check if a new pixel is part of an already fit line segment
  double pxToSegmentDistTh = 1.5;
  // Threshold used to validate the junction jump region. The first eigenvalue of the gradient
  // auto-correlation matrix should be at least junctionEigenvalsTh times bigger than the second eigenvalue
  double junctionEigenvalsTh = 10;
  // the difference between the perpendicular segment direction and the direction of the gradient
  // in the region to be validated must be less than junctionAngleTh radians
  double junctionAngleTh = 10 * (M_PI / 180.0);
  // The threshold over the validation criteria. For ELSED, it is the gradient angular error in pixels.
  double validationTh = 0.15;

  // Whether to validate or not the generated segments
  bool validate = true;
  // Whether to jump over junctions
  bool treatJunctions = true;
  // List of junction size that will be tested (in pixels)
  std::vector<int> listJunctionSizes = {5, 7, 9};
};

/**
 * This class implements the method:
 *     @cite Suárez, I., Buenaposada, J. M., & Baumela, L. (2021).
 *     ELSED: Enhanced Line SEgment Drawing. arXiv preprint arXiv:2108.03144.
 *
 * It is an efficient line segment detector amenable to use in low power devices such as drones or smartphones.
 * The method takes an image as input and outputs a list of detected segments.
 */
class ELSED {
 public:
  // Constructor
  explicit ELSED(const ELSEDParams &params = ELSEDParams());

  /**
   * @brief Detects segments in the input image
   * @param image An input image. The parameters are adapted to images of size 640x480.
   * Bigger images will generate more segments.
   * @return The list of detected segments
   */
  Segments detect(const cv::Mat &image);

  SalientSegments detectSalient(const cv::Mat &image);

  ImageEdges detectEdges(const cv::Mat &image);  // NOLINT

  const LineDetectionExtraInfo &getImgInfo() const;

  const LineDetectionExtraInfoPtr &getImgInfoPtr() const;

  void processImage(const cv::Mat &image);

  void clear();

  static void computeAnchorPoints(const cv::Mat &dirImage,
                                  const cv::Mat &gradImageWO,
                                  const cv::Mat &gradImage,
                                  int scanInterval,
                                  int anchorThresh,
                                  std::vector<Pixel> &anchorPoints);  // NOLINT

  static LineDetectionExtraInfoPtr
  computeGradients(const cv::Mat &srcImg, short gradientTh);

  ImageEdges getAllEdges() const;

  ImageEdges getSegmentEdges() const;

  const EdgeDrawerPtr &getDrawer() const { return drawer; }

 private:
  void drawAnchorPoints(const uint8_t *dirImg,
                        const std::vector<Pixel> &anchorPoints,
                        uint8_t *pEdgeImg);  // NOLINT

  ELSEDParams params;
  LineDetectionExtraInfoPtr imgInfo;
  ImageEdges edges;
  Segments segments;
  SalientSegments salientSegments;
  std::vector<Pixel> anchors;
  EdgeDrawerPtr drawer;
  cv::Mat blurredImg;
  cv::Mat edgeImg;
};
}
#endif //ELSED_ELSED_H_
