#include "ELSED.h"
#include "EdgeDrawer.h"

// Decides if we should take the image gradients as the interpolated version of the pixels right in the segment
// or if they are ready directly from the image
#define UPM_SD_USE_REPROJECTION

namespace upm {

ELSED::ELSED(const ELSEDParams &params) : params(params) {
}

Segments ELSED::detect(const cv::Mat &image) {
  processImage(image);
  // std::cout << "ELSED detected: " << segments.size() << " segments" << std::endl;
  return segments;
}

SalientSegments ELSED::detectSalient(const cv::Mat &image) {
  processImage(image);
  // std::cout << "ELSED detected: " << salientSegments.size() << " salient segments" << std::endl;
  return salientSegments;
}

ImageEdges ELSED::detectEdges(const cv::Mat &image) {
  processImage(image);
  return getSegmentEdges();
}

const LineDetectionExtraInfo &ELSED::getImgInfo() const {
  return *imgInfo;
}

void ELSED::processImage(const cv::Mat &_image) {
  // Check that the image is a grayscale image
  cv::Mat image;
  switch (_image.channels()) {
    case 3:
      cv::cvtColor(_image, image, cv::COLOR_BGR2GRAY);
      break;
    case 4:
      cv::cvtColor(_image, image, cv::COLOR_BGRA2GRAY);
      break;
    default:
      image = _image;
      break;
  }
  assert(image.channels() == 1);
  // Clear previous state
  this->clear();

  if (image.empty()) {
    return;
  }

  // Set the global image
  // Filter the image
  if (params.ksize > 2) {
    cv::GaussianBlur(image, blurredImg, cv::Size(params.ksize, params.ksize), params.sigma);
  } else {
    blurredImg = image;
  }

  // Compute the input image derivatives
  imgInfo = computeGradients(blurredImg, params.gradientThreshold);

  bool anchoThIsZero;
  uint8_t anchorTh = params.anchorThreshold;
  do {
    anchoThIsZero = anchorTh == 0;
    // Detect edges and segment in the input image
    computeAnchorPoints(imgInfo->dirImg,
                        imgInfo->gImgWO,
                        imgInfo->gImg,
                        params.scanIntervals,
                        anchorTh,
                        anchors);

    // If we couldn't find any anchor, decrease the anchor threshold
    if (anchors.empty()) {
      // std::cout << "Cannot find any anchor with AnchorTh = " << int(anchorTh)
      //      << ", repeating process with AnchorTh = " << (anchorTh / 2) << std::endl;
      anchorTh /= 2;
    }

  } while (anchors.empty() && !anchoThIsZero);
  // LOGD << "Detected " << anchors.size() << " anchor points ";
  edgeImg = cv::Mat::zeros(imgInfo->imageHeight, imgInfo->imageWidth, CV_8UC1);
  drawer = std::make_shared<EdgeDrawer>(imgInfo,
                                        edgeImg,
                                        params.lineFitErrThreshold,
                                        params.pxToSegmentDistTh,
                                        params.minLineLen,
                                        params.treatJunctions,
                                        params.listJunctionSizes,
                                        params.junctionEigenvalsTh,
                                        params.junctionAngleTh);

  drawAnchorPoints(imgInfo->dirImg.ptr(), anchors, edgeImg.ptr());
}

LineDetectionExtraInfoPtr ELSED::computeGradients(const cv::Mat &srcImg, short gradientTh) {
  LineDetectionExtraInfoPtr dstInfo = std::make_shared<LineDetectionExtraInfo>();
  cv::Sobel(srcImg, dstInfo->dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(srcImg, dstInfo->dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

  int nRows = srcImg.rows;
  int nCols = srcImg.cols;
  int i;

  dstInfo->imageWidth = srcImg.cols;
  dstInfo->imageHeight = srcImg.rows;
  dstInfo->gImgWO = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->gImg = cv::Mat(srcImg.size(), dstInfo->dxImg.type());
  dstInfo->dirImg = cv::Mat(srcImg.size(), CV_8UC1);

  const int16_t *pDX = dstInfo->dxImg.ptr<int16_t>();
  const int16_t *pDY = dstInfo->dyImg.ptr<int16_t>();
  auto *pGr = dstInfo->gImg.ptr<int16_t>();
  auto *pGrWO = dstInfo->gImgWO.ptr<int16_t>();
  auto *pDir = dstInfo->dirImg.ptr<uchar>();
  int16_t abs_dx, abs_dy, sum;
  const int totSize = nRows * nCols;
  for (i = 0; i < totSize; ++i) {
    // Absolute value
    abs_dx = UPM_ABS(pDX[i]);
    // Absolute value
    abs_dy = UPM_ABS(pDY[i]);
    sum = abs_dx + abs_dy;
    // Divide by 2 the gradient
    pGrWO[i] = sum;
    pGr[i] = sum < gradientTh ? 0 : sum;
    // Select between vertical or horizontal gradient
    pDir[i] = abs_dx >= abs_dy ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;
  }

  return dstInfo;
}

inline void ELSED::computeAnchorPoints(const cv::Mat &dirImage,
                                       const cv::Mat &gradImageWO,
                                       const cv::Mat &gradImage,
                                       int scanInterval,
                                       int anchorThresh,
                                       std::vector<Pixel> &anchorPoints) {  // NOLINT

  int imageWidth = gradImage.cols;
  int imageHeight = gradImage.rows;

  // Get pointers to the thresholded gradient image and to the direction image
  const auto *gradImg = gradImage.ptr<int16_t>();
  const auto *dirImg = dirImage.ptr<uint8_t>();

  // Extract the anchors in the gradient image, store into a vector
  unsigned int pixelNum = imageWidth * imageHeight;
  unsigned int edgePixelArraySize = pixelNum / (2.5 * scanInterval);
  anchorPoints.resize(edgePixelArraySize);

  int nAnchors = 0;
  int indexInArray;
  unsigned int w, h;
  for (w = 1; w < imageWidth - 1; w += scanInterval) {
    for (h = 1; h < imageHeight - 1; h += scanInterval) {
      indexInArray = h * imageWidth + w;

      // If there is no gradient in the pixel avoid the anchor generation
      if (gradImg[indexInArray] == 0) continue;

      // To be an Anchor the pixel must have a gradient magnitude
      // anchorThreshold_ units higher than that of its neighbours
      if (dirImg[indexInArray] == UPM_EDGE_HORIZONTAL) {
        // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
        // We compare with the top and bottom pixel gradients
        if (gradImg[indexInArray] >= gradImg[indexInArray - imageWidth] + anchorThresh &&
            gradImg[indexInArray] >= gradImg[indexInArray + imageWidth] + anchorThresh) {
          anchorPoints[nAnchors].x = w;
          anchorPoints[nAnchors].y = h;
          nAnchors++;
        }
      } else {
        // Check if (w, h) is accepted as an anchor using the Anchor Threshold.
        // We compare with the left and right pixel gradients
        if (gradImg[indexInArray] >= gradImg[indexInArray - 1] + anchorThresh &&
            gradImg[indexInArray] >= gradImg[indexInArray + 1] + anchorThresh) {
          anchorPoints[nAnchors].x = w;
          anchorPoints[nAnchors].y = h;
          nAnchors++;
        }
      }
    }
  }
  anchorPoints.resize(nAnchors);
}

void ELSED::clear() {
  imgInfo = nullptr;
  edges.clear();
  segments.clear();
  salientSegments.clear();
  anchors.clear();
  drawer = nullptr;
  blurredImg = cv::Mat();
  edgeImg = cv::Mat();
}

inline int calculateNumPtsToTrim(int nPoints) {
  return std::min(5.0, nPoints * 0.1);
}

// Linear interpolation. s is the starting value, e the ending value
// and t the point offset between e and s in range [0, 1]
inline float lerp(float s, float e, float t) { return s + (e - s) * t; }

// Bi-linear interpolation of point (tx, ty) in the cell with corner values [[c00, c01], [c10, c11]]
inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
  return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void ELSED::drawAnchorPoints(const uint8_t *dirImg,
                             const std::vector<Pixel> &anchorPoints,
                             uint8_t *pEdgeImg) {
  assert(imgInfo && imgInfo->imageWidth > 0 && imgInfo->imageHeight > 0);
  assert(!imgInfo->gImg.empty() && !imgInfo->dirImg.empty() && pEdgeImg);
  assert(drawer);
  assert(!edgeImg.empty());

  int imageWidth = imgInfo->imageWidth;
  int imageHeight = imgInfo->imageHeight;
  bool expandHorizontally;
  int indexInArray;
  unsigned char lastDirection;  // up = 1, right = 2, down = 3, left = 4;

  if (anchorPoints.empty()) {
    // No anchor points detected in the image
    return;
  }

  const double validationTh = params.validationTh;

  for (const auto &anchorPoint: anchorPoints) {
    // LOGD << "Managing new Anchor point: " << anchorPoint;
    indexInArray = anchorPoint.y * imageWidth + anchorPoint.x;

    if (pEdgeImg[indexInArray]) {
      // If anchor i is already been an edge pixel
      continue;
    }

    // If the direction of this pixel is horizontal, then go left and right.
    expandHorizontally = dirImg[indexInArray] == UPM_EDGE_HORIZONTAL;

    /****************** First side Expanding (Right or Down) ***************/
    // Select the first side towards we want to move. If the gradient points
    // horizontally select the right direction and otherwise the down direction.
    lastDirection = expandHorizontally ? UPM_RIGHT : UPM_DOWN;

    drawer->drawEdgeInBothDirections(lastDirection, anchorPoint);
  }

  double theta, angle;
  float saliency;
  bool valid;
  int endpointDist, nOriInliers, nOriOutliers;
#ifdef UPM_SD_USE_REPROJECTION
  cv::Point2f p;
  float lerp_dx, lerp_dy;
  int x0, y0, x1, y1;
#endif
  int16_t *pDx = imgInfo->dxImg.ptr<int16_t>();
  int16_t *pDy = imgInfo->dyImg.ptr<int16_t>();
  segments.reserve(drawer->getDetectedFullSegments().size());
  salientSegments.reserve(drawer->getDetectedFullSegments().size());

  for (const FullSegmentInfo &detectedSeg: drawer->getDetectedFullSegments()) {

    valid = true;
    if (params.validate) {
      if (detectedSeg.getNumOfPixels() < 2) {
        valid = false;
      } else {
        // Get the segment angle
        Segment s = detectedSeg.getEndpoints();
        theta = segAngle(s) + M_PI_2;
        // Force theta to be in range [0, M_PI)
        while (theta < 0) theta += M_PI;
        while (theta >= M_PI) theta -= M_PI;

        // Calculate the line equation as the cross product os the endpoints
        cv::Vec3f l = cv::Vec3f(s[0], s[1], 1).cross(cv::Vec3f(s[2], s[3], 1));
        // Normalize the line direction
        l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
        cv::Point2f perpDir(l[0], l[1]);

        // For each pixel in the segment compute its angle
        int nPixelsToTrim = calculateNumPtsToTrim(detectedSeg.getNumOfPixels());

        Pixel firstPx = detectedSeg.getFirstPixel();
        Pixel lastPx = detectedSeg.getLastPixel();

        nOriInliers = 0;
        nOriOutliers = 0;

        for (auto px: detectedSeg) {

          // If the point is not an inlier avoid it
          if (edgeImg.at<uint8_t>(px.y, px.x) != UPM_ED_SEGMENT_INLIER_PX) {
            continue;
          }

          endpointDist = detectedSeg.horizontal() ?
                         std::min(std::abs(px.x - lastPx.x), std::abs(px.x - firstPx.x)) :
                         std::min(std::abs(px.y - lastPx.y), std::abs(px.y - firstPx.y));

          if (endpointDist < nPixelsToTrim) {
            continue;
          }

#ifdef UPM_SD_USE_REPROJECTION
          // Re-project the point into the segment. To do this, we should move pixel.dot(l)
          // units (the distance between the pixel and the segment) in the direction
          // perpendicular to the segment (perpDir).
          p = cv::Point2f(px.x, px.y) - perpDir * cv::Vec3f(px.x, px.y, 1).dot(l);
          // Get the values around the point p to do the bi-linear interpolation
          x0 = p.x < 0 ? 0 : p.x;
          if (x0 >= imageWidth) x0 = imageWidth - 1;
          y0 = p.y < 0 ? 0 : p.y;
          if (y0 >= imageHeight) y0 = imageHeight - 1;
          x1 = p.x + 1;
          if (x1 >= imageWidth) x1 = imageWidth - 1;
          y1 = p.y + 1;
          if (y1 >= imageHeight) y1 = imageHeight - 1;
          //Bi-linear interpolation of Dx and Dy
          lerp_dx = blerp(pDx[y0 * imageWidth + x0], pDx[y0 * imageWidth + x1],
                          pDx[y1 * imageWidth + x0], pDx[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          lerp_dy = blerp(pDy[y0 * imageWidth + x0], pDy[y0 * imageWidth + x1],
                          pDy[y1 * imageWidth + x0], pDy[y1 * imageWidth + x1],
                          p.x - int(p.x), p.y - int(p.y));
          // Get the gradient angle
          angle = std::atan2(lerp_dy, lerp_dx);
#else
          indexInArray = px.y * imageWidth + px.x;
          angle = std::atan2(pDy[indexInArray], pDx[indexInArray]);
#endif
          // Force theta to be in range [0, M_PI)
          if (angle < 0) angle += M_PI;
          if (angle >= M_PI) angle -= M_PI;
          circularDist(theta, angle, M_PI) > validationTh ? nOriOutliers++ : nOriInliers++;
        }

        valid = nOriInliers > nOriOutliers;
        saliency = nOriInliers;
      }
    } else {
      saliency = segLength(detectedSeg.getEndpoints());
    }
    if (valid) {
      const Segment &endpoints = detectedSeg.getEndpoints();
      segments.push_back(endpoints);
      salientSegments.emplace_back(endpoints, saliency);
    }
  }
}

ImageEdges ELSED::getAllEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

ImageEdges ELSED::getSegmentEdges() const {
  assert(drawer);
  ImageEdges result;
  for (const FullSegmentInfo &s: drawer->getDetectedFullSegments())
    result.push_back(s.getPixels());
  return result;
}

const LineDetectionExtraInfoPtr &ELSED::getImgInfoPtr() const {
  return imgInfo;
}

}  // namespace upm
