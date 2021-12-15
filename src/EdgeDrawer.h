#ifndef ELSED_EDGEDRAWER_H_
#define ELSED_EDGEDRAWER_H_

#include <stack>
#include <memory>
#include "Utils.h"
#include "FullSegmentInfo.h"

namespace upm {

struct DrawingBranch {
  uint8_t direction;
  Pixel px;
  bool addPixelsForTheFirstSide;
  ImageEdge pixels;
};

class EdgeDrawer {
 public:

  EdgeDrawer() = default;

  EdgeDrawer(const LineDetectionExtraInfoPtr &gradientInfo,
             cv::Mat &edgeImage,
             float lineFitThreshold = 0.2,
             float pxToSegmentDistTh = 1.5,
             int minLineLength = 15,
             bool treatJunctions = true,
             std::vector<int> mListOfJunctionSizes = {5, 7, 9},
             double junctionEigenvalsTh = 10,
             double junctionAngleTh = 10 * (M_PI / 180.0));

  void init(const LineDetectionExtraInfoPtr &gradientInfo, cv::Mat &edgeImage);

  void drawEdgeInBothDirections(uint8_t direction, Pixel anchor);

  inline const std::vector<FullSegmentInfo> &getDetectedFullSegments() const {
    return segments;
  }

  Segments getDetectedSegments() const;

  const ImageEdge &getPixels() const { return pixels; }

 private:
  inline void addJunctionPixelsToSegment(const ImageEdge &edgePixels,
                                         FullSegmentInfo &segment,
                                         bool addPixelsForTheFirstSide);

  /**
   *
   * @param gradImg
   * @param eq
   * @param invertLineDir
   * @param imageWidth
   * @param imageHeight
   * @param stepSize The number of pixels we should skip.
   * @param px
   * @return
   */
  inline static bool findNextPxWithProjection(const int16_t *gradImg,
                                              cv::Vec3f eq,
                                              bool invertLineDir,  // extendSecondEndpoint
                                              int imageWidth, int imageHeight,
                                              int stepSize,
                                              Pixel &px);

  inline static bool findNextPxWithGradient(uint8_t pxGradDirection,  // dirImg[indexInArray]
                                            const int16_t *gradImg,
                                            int imageWidth, int imageHeight,
                                            Pixel &px,
                                            Pixel &lastPx);

  /**
   * @brief Checks if a segment can be extended by a certain side.
   * The criteria to know if a segment can be extended is to walk
   * UPM_MAX_OUTLIERS_TH pixels in the edge direction and check if
   * leaving free the drawing algorithm, the segment can be extended, i.e. that
   * the ED method can find other UPM_MAX_OUTLIERS_TH inliers in that direction.
   *
   * @param segment
   * @param extendByTheEnd
   * @param gradImg
   * @param imageWidth
   * @param imageHeight
   * @param lineFitThreshold
   * @param pixelsInTheExtension
   * @return
   */
  bool canSegmentBeExtended(FullSegmentInfo &segment,
                            bool extendByTheEnd,
                            ImageEdge &pixelsInTheExtension);

  static inline uint8_t inverseDirection(uint8_t dir);

  static inline uint8_t directionFromLineEq(const cv::Vec3d &vec);

  /**
   * 
   * @param anchor 
   * @param initialPixels Output pixels to be used in the opposite direction of the anchor.
   */
  void drawEdgeTreeStack(Pixel anchor, ImageEdge &initialPixels, bool firstAnchorDirection);

  std::vector<FullSegmentInfo> segments;
  std::vector<DrawingBranch> branchesStack;
  ImageEdge pixels;
  const int16_t *gradImg;
  const int16_t *pDxImg;
  const int16_t *pDyImg;
  const uint8_t *pDirImg;
  uint8_t *edgeImg;
  int imageWidth, imageHeight;
  float lineFitThreshold = 0.2;
  float pxToSegmentDistTh = 1.5;
  int minLineLength = 15;
  bool treatJunctions = true;
  std::vector<int> listOfJunctionSizes = {5, 7, 9};
  double junctionEigenvalsTh = 10;
  double junctionAngleTh = 10 * (M_PI / 180.0);
};

typedef std::shared_ptr<EdgeDrawer> EdgeDrawerPtr;
}  // namespace upm

#endif //ELSED_EDGEDRAWER_H_
