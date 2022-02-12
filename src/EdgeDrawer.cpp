#include <utility>
#include <memory>
#include "EdgeDrawer.h"

namespace upm {

template<class T>
static inline std::vector<T> &
operator+=(std::vector<T> &lhs, std::vector<T> l) {
  lhs.insert(std::end(lhs), std::begin(l), std::end(l));
  return lhs;
}

EdgeDrawer::EdgeDrawer(const LineDetectionExtraInfoPtr &gradientInfo,
                       cv::Mat &edgeImage,
                       float lineFitThreshold,
                       float pxToSegmentDistTh,
                       int minLineLength,
                       bool treatJunctions,
                       std::vector<int> listOfJunctionSizes,
                       double junctionEigenvalsTh,
                       double junctionAngleTh) :
    lineFitThreshold(lineFitThreshold),
    pxToSegmentDistTh(pxToSegmentDistTh),
    minLineLength(minLineLength),
    treatJunctions(treatJunctions),
    listOfJunctionSizes(std::move(listOfJunctionSizes)),
    junctionEigenvalsTh(junctionEigenvalsTh),
    junctionAngleTh(junctionAngleTh) {
  init(gradientInfo, edgeImage);
}

void EdgeDrawer::init(const LineDetectionExtraInfoPtr &gradientInfo, cv::Mat &edgeImage) {
  assert(gradientInfo->imageWidth == gradientInfo->gImg.cols);
  assert(gradientInfo->imageHeight == gradientInfo->gImg.rows);
  assert(gradientInfo->imageWidth > 0 && gradientInfo->imageHeight > 0);
  imageWidth = gradientInfo->imageWidth;
  imageHeight = gradientInfo->imageHeight;

  assert(!gradientInfo->gImg.empty());
  assert(gradientInfo->gImg.type() == CV_16SC1);
  gradImg = gradientInfo->gImg.ptr<int16_t>();
  assert(!gradientInfo->dirImg.empty());
  assert(gradientInfo->dirImg.type() == CV_8UC1);
  assert(gradientInfo->dirImg.size() == gradientInfo->gImg.size());
  pDirImg = gradientInfo->dirImg.ptr<uint8_t>();
  assert(!gradientInfo->dxImg.empty());
  assert(gradientInfo->dxImg.type() == CV_16SC1);
  assert(gradientInfo->dxImg.size() == gradientInfo->gImg.size());
  pDxImg = gradientInfo->dxImg.ptr<int16_t>();
  assert(!gradientInfo->dyImg.empty());
  assert(gradientInfo->dyImg.type() == CV_16SC1);
  assert(gradientInfo->dyImg.size() == gradientInfo->gImg.size());
  pDyImg = gradientInfo->dyImg.ptr<int16_t>();

  assert(!edgeImage.empty());
  assert(edgeImage.type() == CV_8UC1);
  assert(edgeImage.size() == gradientInfo->gImg.size());
  edgeImg = edgeImage.ptr<uint8_t>();

  pixels.reserve(0.25 * imageWidth * imageHeight);
  segments.reserve(imageWidth / 2);
  branchesStack.reserve(100);
}

inline void EdgeDrawer::addJunctionPixelsToSegment(const ImageEdge &junctionPixels,
                                                   FullSegmentInfo &segment,
                                                   bool addPixelsForTheFirstSide) {

  if (addPixelsForTheFirstSide) segment.firstEndpointExtended = true;
  else segment.secondEndpointExtended = true;

  // Mark the pixels as inliers or outliers and add it to the segment
  for (int i = 0; i < junctionPixels.size(); i++) {
    const Pixel &junctionPx = junctionPixels[i];
    if (segment.isInlier(junctionPx.x, junctionPx.y, pxToSegmentDistTh)) {
      if (i < junctionPixels.size() / 2) {
        edgeImg[junctionPx.y * imageWidth + junctionPx.x] = UPM_ED_JUNTION_PX; // UPM_ED_SEGMENT_INLIER_PX;
      } else {
        edgeImg[junctionPx.y * imageWidth + junctionPx.x] = UPM_ED_SEGMENT_INLIER_PX;
      }
      pixels.push_back(junctionPx);
      segment.addPixel(junctionPx.x, junctionPx.y, pixels.size() - 1, addPixelsForTheFirstSide);
    } else {
      edgeImg[junctionPx.y * imageWidth + junctionPx.x] = UPM_ED_SEGMENT_OUTLIER_PX;
    }
  }
}

inline Pixel calcLastPixelWithDirection(const Pixel &px, uint8_t lastDir) {
  // If there is no last pixel, calculate if from the lastDirection
  switch (lastDir) {
    case UPM_UP:return {px.x, px.y + 1};
    case UPM_DOWN:return {px.x, px.y - 1};
    case UPM_LEFT:return {px.x + 1, px.y};
    case UPM_RIGHT:return {px.x - 1, px.y};
    default: return {-1, -1};
  }
}

void EdgeDrawer::drawEdgeTreeStack(Pixel anchor, ImageEdge &initialPixels, bool firstAnchorDirection) {

  // The index in the pixels array of the last segment we have tried to add
  uint8_t direction, gradDir, lineDirection;
  int i, indexInArray, lastChekedPxIdx, initialPxIndex, indexInImage, nElements;
  bool addPixelsForTheFirstSide, inlierOverwritten, popStack, firstBranch, isAnchorFirstPx, wasExtended;
  bool localSegInitialized, segment;
  Pixel px, lastPx;
  double fitError;
  ImageEdge extensionPixels, outliersList;

  FullSegmentInfo *localSegment = nullptr;  // (pixels, nullptr);
  extensionPixels.reserve(minLineLength);
  // Reserve a big space to store the edge pixels

  segment = false;
  firstBranch = true;
  // Check if we are extending an anchor by its second side
  isAnchorFirstPx = pixels.back() == anchor && firstAnchorDirection;

  segments.emplace_back(pixels);
  localSegment = &segments.back();

  while (!branchesStack.empty()) {
    popStack = true;
    DrawingBranch &branch = branchesStack.back();
    // The index of the first pixel that is part of the branch
    initialPxIndex = pixels.size();
    // If we are extending an anchor by its second side, start one pixel before
    if (firstBranch && isAnchorFirstPx) initialPxIndex--;
    direction = branch.direction;
    px = branch.px;
    addPixelsForTheFirstSide = branch.addPixelsForTheFirstSide;
    pixels += branch.pixels;

//    LOGD << " ---- STARTING NEW DRAWING CALL ----\n\t- px: " << px << "\n\t- direction: " << dirToStr(direction)
//         << "\n\t- addPixelsForTheFirstSide: " << addPixelsForTheFirstSide << "\n\t- segment: " << segment
//         << "\n\t- Initial Pixels: " << branch.pixels;

    // The index in the pixels array of the last segment we have tried to add
    lastChekedPxIdx = initialPxIndex;
    // The current number of inliers
    outliersList.clear();
    inlierOverwritten = false;

    lastPx = calcLastPixelWithDirection(px, direction);

    gradDir = (direction == UPM_DOWN || direction == UPM_UP) ? UPM_EDGE_VERTICAL : UPM_EDGE_HORIZONTAL;

    localSegInitialized = false;
    while (outliersList.size() <= UPM_MAX_OUTLIERS_TH) {

      if (!findNextPxWithGradient(gradDir, gradImg, imageWidth, imageHeight, px, lastPx)) {
        // Stopping because the gradient level is 0 or the pixel is out of the image
        break;
      }

      indexInArray = px.y * imageWidth + px.x;
      inlierOverwritten = edgeImg[indexInArray] == UPM_ED_SEGMENT_INLIER_PX ||
          edgeImg[indexInArray] == UPM_ED_JUNTION_PX ||
          edgeImg[indexInArray] == UPM_ED_EDGE_PIXEL;

      if (inlierOverwritten) {
        // LOGD << "\t\tStopping because the pixel " << px << " is an "
        //      << (edgeImg[indexInArray] == UPM_ED_SEGMENT_INLIER_PX ? "INLIER" : "EDGE_PIXEL")
        //      << " and inlierOverwritten = " << inlierOverwritten;
        break;
      }

      // LOGD << "\tDrawn new pixel " << px;
      // If we are following pixels that lie in a segment
      if (segment) {

        if (localSegment->isInlier(px.x, px.y, pxToSegmentDistTh)) {
          // Mark the pixel as edge only if it is an inlier
          edgeImg[indexInArray] = UPM_ED_SEGMENT_INLIER_PX;
          // LOGD << "\t\tIt's an INLIER! :)";

          // Add the next pixel of the edge to the vector
          pixels.push_back(px);

          if (!outliersList.empty()) {
            // Add the information of the number of outliers to the segment
            // Reset the number of outliers to 0
            outliersList.clear();
          }

          // The pixel is part of the segment, so add it
          localSegment->addPixel(px.x, px.y, pixels.size() - 1, addPixelsForTheFirstSide);

          // Mark the current side as ready to be extended again
          if (addPixelsForTheFirstSide) localSegment->firstEndpointExtended = false;
          else localSegment->secondEndpointExtended = false;

        } else {
          // LOGD << "\t\tIt's an OUTLIER! :(";
          outliersList.push_back(px);
          edgeImg[indexInArray] = UPM_ED_SEGMENT_OUTLIER_PX;
        }
      } else {  // !segment: We have not yet adjusted a segment to the pixels

        // Add the next pixel of the edge to the vector
        pixels.push_back(px);

        // Mark the pixel as edge only if it is an inlier
        edgeImg[indexInArray] = UPM_ED_EDGE_PIXEL;

        // If there are enough pixels try to fit a segment to them
        if (pixels.size() - lastChekedPxIdx >= minLineLength) {
          //If we have already set the first minLineLength pixels, just add the new UPM_SKIP_EDGE_PT ones
          if (localSegInitialized) {
            localSegment->skipPositions();
          } else {
            // If have never tried to fit a segment, do it now
            localSegInitialized = true;
            localSegment->init(pixels, lastChekedPxIdx);
          }

          fitError = localSegment->getFitError();
          // LOGD << "\t\tTrying to fit a segment, fitError: " << fitError;

          if (fitError < lineFitThreshold) {
            segment = true;
            for (i = lastChekedPxIdx; i < lastChekedPxIdx + minLineLength; i++) {
              edgeImg[pixels[i].y * imageWidth + pixels[i].x] = UPM_ED_SEGMENT_INLIER_PX;
            }
          } else {
            // If the fitting error has increased more than lineFitThreshold, release the segment
            lastChekedPxIdx += UPM_SKIP_EDGE_PT;
          }
        }
      }

      // Update the gradient direction with the selected pixel
      gradDir = pDirImg[indexInArray];

    }  // End of while, we have finished extending this edge

    // LOGD << "Drawing process has ended " << (segment ? "FOUND ONE SEGMENT" : "WITHOUT SEGMENTS");

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////


    // Save the pixels next to the anchor point to try to extend it later
    if (firstBranch && edgeImg[anchor.y * imageWidth + anchor.x] != UPM_ED_SEGMENT_INLIER_PX) {
      // Take the last minLineLength pixels and add it to the new edge branch
      initialPixels.clear();
      nElements = std::min(int(pixels.size() - initialPxIndex), (minLineLength - 1));
      for (i = initialPxIndex + nElements - 1; i >= initialPxIndex; i--) {

        // If the pixel is part of some segment
        indexInImage = pixels[i].y * imageWidth + pixels[i].x;
        if (edgeImg[indexInImage] == UPM_ED_SEGMENT_INLIER_PX || edgeImg[indexInImage] == UPM_ED_JUNTION_PX)
          break;
        // Add the pixel to the new branch
        initialPixels.push_back(pixels[i]);
      }
    }

    // If (!segment) Do nothing
    // If a segment was detected at the moment we finish the search process
    if (segment) {
      if (!outliersList.empty()) {
        // Remove the last pixel
        if (addPixelsForTheFirstSide || localSegment->hasSecondSideElements()) {
          Pixel pxToDelete = addPixelsForTheFirstSide ? localSegment->getLastPixel() : localSegment->getFirstPixel();
          edgeImg[pxToDelete.y * imageWidth + pxToDelete.x] = UPM_ED_NO_EDGE_PIXEL;
          localSegment->removeLastPx(addPixelsForTheFirstSide);

          // Remove the last pixel from pixels
          outliersList.insert(outliersList.begin(), pixels.back());
          pixels.pop_back();

          if (!addPixelsForTheFirstSide && localSegment->getFirstPixel() == anchor) {
            // If we have removed the only pixel that was left at the initial part of the segment,
            // we mark that part as already extended
            localSegment->secondEndpointExtended = true;
          }
        }
      }

      // BRANCH 3
      if (outliersList.size() > UPM_MAX_OUTLIERS_TH) {
        // If outliersList.size() > UPM_MAX_OUTLIERS_TH, we generate a THIRD BRANCH to draw without following a segment
        // This is not immediately processed, instead, it is added to the stack and processed
        // when the previous tasks have finished
        for (const Pixel &outlier: outliersList) {
          edgeImg[outlier.y * imageWidth + outlier.x] = UPM_ED_NO_EDGE_PIXEL;
        }

        // Continue drawing in the gradient direction
        uint8_t predictedLastDir;
        if (pDirImg[indexInArray] == UPM_EDGE_HORIZONTAL) predictedLastDir = lastPx.x < px.x ? UPM_RIGHT : UPM_LEFT;
        else predictedLastDir = lastPx.y < px.y ? UPM_DOWN : UPM_UP;

        popStack = false;
        branch.direction = predictedLastDir;
        branch.px.x = px.x;
        branch.px.y = px.y;
        branch.addPixelsForTheFirstSide = true;

        // branch.pixels should contain the last inlier px and the list of outliers
        branch.pixels.clear();
        branch.pixels.push_back(pixels.back());
        for (Pixel &p: outliersList) branch.pixels.push_back(p);
      }

      // BRANCH 1
      wasExtended = localSegment->firstEndpointExtended;
      // If we can go on straight forward skipping some pixels do it
      if ((outliersList.size() > UPM_MAX_OUTLIERS_TH || inlierOverwritten) &&
          !wasExtended && treatJunctions && canSegmentBeExtended(*localSegment,
                                                                 true,
                                                                 extensionPixels)) {
        // Generating FIRST BRANCH to continue straight forward

        // Get the line direction
        lineDirection = directionFromLineEq(localSegment->getLineEquation());
        // Add the junction pixels to the segment
        addJunctionPixelsToSegment(extensionPixels, *localSegment, addPixelsForTheFirstSide);

        if (popStack) {
          // Re-use the last entry of the stack
          popStack = false;
          branch.direction = lineDirection;
          branch.px = extensionPixels.back();
          branch.addPixelsForTheFirstSide = true;
          branch.pixels.clear();

        } else {
          // Create a new entry in the stack
          branchesStack.emplace_back(DrawingBranch{lineDirection, extensionPixels.back(), true, {}});
        }

      } else {
        // BRANCH 2 (a, b)
        // Go in the opposite line direction extending the other side of the segment
        localSegment->firstEndpointExtended = true;

        // If we have found a segment, try to extend the other side of it
        wasExtended = localSegment->secondEndpointExtended;
        // Generate the second branch only if we have completely generated the first one
        if (!wasExtended && localSegment->getFirstPixel() == anchor) {
          // BRANCH 2.a
          // Generating SECOND BRANCH (a) in the second segment direction

          // If we are dealing with the segment that contains the anchor point,
          // do not skip pixels to expand the second part

          // Get the line direction
          uint8_t oppositeLineDirection = inverseDirection(directionFromLineEq(localSegment->getLineEquation()));
          localSegment->secondEndpointExtended = true;

          if (popStack) {
            // Re-use the last entry of the stack
            popStack = false;
            branch.direction = oppositeLineDirection;
            branch.px = anchor;
            branch.addPixelsForTheFirstSide = false;
            branch.pixels.clear();
          } else {
            // Create a new entry in the stack
            branchesStack.emplace_back(DrawingBranch{oppositeLineDirection, anchor, false, {}});
          }
        } else if (!wasExtended && treatJunctions && canSegmentBeExtended(*localSegment,
                                                                          false,
                                                                          extensionPixels)) {

          // BRANCH 2.b
          // Generating SECOND BRANCH (b) with extension pixels (JUNCTION)
          // Add the junction pixels to the segment
          addJunctionPixelsToSegment(extensionPixels, *localSegment, false);

          // Get the line direction
          uint8_t opositeLineDirection = inverseDirection(directionFromLineEq(localSegment->getLineEquation()));

          if (popStack) {
            // Re-use the last entry of the stack
            popStack = false;
            branch.direction = opositeLineDirection;
            branch.px = extensionPixels.back();
            branch.addPixelsForTheFirstSide = false;
            branch.pixels.clear();
          } else {
            // Create a new entry in the stack
            branchesStack.emplace_back(DrawingBranch{opositeLineDirection, extensionPixels.back(), false, {}});
          }
        } else {

          // NO BRANCH 2 NOR 1
          // LOGD << "Do NOT Generate SECOND BRANCH (wasExtended = " << wasExtended << ")";
          localSegment->secondEndpointExtended = true;
          // The segment was completely extended so save it
          localSegment->finish();
          // create a new segment if we are not in the last loop
          segments.emplace_back(pixels);
          localSegment = &segments.back();
          segment = false;
        }
      }
    }

    if (popStack) {
      // This stack only do pop if no new branch has been added
      branchesStack.resize(branchesStack.size() - 1);
    }

    firstBranch = false;
    extensionPixels.clear();

  }  // End of while(!mBranches.empthy())

  // Remove the last segment that is not valid
  segments.pop_back();

}  // End of method

void EdgeDrawer::drawEdgeInBothDirections(uint8_t direction, Pixel anchor) {
  assert(anchor.x >= 0 && anchor.x < imageWidth && anchor.y >= 0 && anchor.y < imageHeight);
  assert(gradImg && gradImg[anchor.y * imageWidth + anchor.x] > 0);

  ImageEdge anchorPixels, newBranchPixels;
  edgeImg[anchor.y * imageWidth + anchor.x] = UPM_ED_ANCHOR_PIXEL;

  // Continue drawing in the gradient direction
  // LOGD << "\tExtending anchor: " << anchor << " in FIRST direction: " << dirToStr(direction);
  branchesStack.emplace_back(DrawingBranch{direction, anchor, true, {}});
  pixels.push_back(anchor);
  drawEdgeTreeStack(anchor, anchorPixels, true);

  uint8_t oppositeDirection = inverseDirection(direction);
  branchesStack.emplace_back(DrawingBranch{oppositeDirection, anchor, true, anchorPixels});
  drawEdgeTreeStack(anchor, anchorPixels, false);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

inline bool EdgeDrawer::findNextPxWithGradient(uint8_t pxGradDirection,  // dirImg[indexInArray]
                                               const int16_t *gradImg,
                                               int imageWidth, int imageHeight,
                                               Pixel &px,
                                               Pixel &lastPx) {

  int16_t gValue1, gValue2, gValue3;
  int lastX, lastY, indexInArray;

  indexInArray = px.y * imageWidth + px.x;
  lastX = lastPx.x;
  lastY = lastPx.y;
  lastPx = px;

  if (pxGradDirection == UPM_EDGE_HORIZONTAL) {
    // The image gradient points horizontally so lets check:
    // ---------------
    // | Ok | X | Ok |
    // | Ok | X | Ok |
    // | Ok | X | Ok |
    // ---------------
    if (lastX < px.x) {
      // ---------------
      // | X | X | Ok |
      // | X | X | Ok |
      // | X | X | Ok |
      // ---------------
      // Go right
      if (px.x == imageWidth - 1 || px.y == 0 || px.y == imageHeight - 1) {
        // Reach the image border
        return false;
      }

      indexInArray++;
      px.x++;
      gValue1 = gradImg[indexInArray - imageWidth];
      gValue2 = gradImg[indexInArray];
      gValue3 = gradImg[indexInArray + imageWidth];
      if (gValue1 > gValue2 && gValue1 > gValue3) {
        // Up-right
        px.y = px.y - 1;
      } else if (gValue3 > gValue2 && gValue3 > gValue1) {
        // Down-right
        px.y = px.y + 1;
      } // else Straight-right

    } else if (lastX > px.x) {
      // ---------------
      // | Ok | X | X |
      // | Ok | X | X |
      // | Ok | X | X |
      // ---------------
      // Go left
      if (px.x == 0 || px.y == 0 || px.y == imageHeight - 1) {
        // Reach the image border
        return false;
      }

      indexInArray--;
      px.x--;
      gValue1 = gradImg[indexInArray - imageWidth];
      gValue2 = gradImg[indexInArray];
      gValue3 = gradImg[indexInArray + imageWidth];
      if (gValue1 > gValue2 && gValue1 > gValue3) {
        // Up-left
        px.y = px.y - 1;
      } else if (gValue3 > gValue2 && gValue3 > gValue1) {
        // Down-left
        px.y = px.y + 1;
      }

    } else if (lastY < px.y) { // lastX == px.x
      // ---------------
      // | X  | X | X  |
      // | X  | X | X  |
      // | Ok | X | Ok |
      // ---------------
      if (px.y == imageHeight - 1 || px.x == 0 || px.x == imageWidth - 1) {
        // Reach the image border
        return false;
      }

      indexInArray += imageWidth;
      px.y++;
      if (gradImg[indexInArray - 1] > gradImg[indexInArray + 1]) {
        // Dow-left
        px.x--;
      } else {
        // Down-right
        px.x++;
      } // else straight down

    } else { // lastX == px.x && lastY > px.y
      // ---------------
      // | Ok | X | Ok |
      // | X  | X | X  |
      // | X  | X | X  |
      // ---------------
      if (px.y == 0 || px.x == 0 || px.x == imageWidth - 1) {
        // Reach the image border
        return false;
      }

      indexInArray -= imageWidth;
      px.y--;
      if (gradImg[indexInArray - 1] > gradImg[indexInArray + 1]) {
        // Up-left
        px.x--;
      } else {
        // Up-right
        px.x++;
      }
    }

  } else {
    // The image gradient points vertically so lets check:
    // ----------------
    // | Ok | Ok | Ok |
    // |  X |  X |  X |
    // | Ok | Ok | Ok |
    // ----------------
    if (lastY < px.y) {
      // ----------------
      // |  X |  X |  X |
      // |  X |  X |  X |
      // | Ok | Ok | Ok |
      // ----------------
      // Go down
      if (px.y == imageHeight - 1 || px.x == 0 || px.x == imageWidth - 1) {
        // Reach the image border
        return false;
      }

      indexInArray += imageWidth;
      px.y++;
      gValue1 = gradImg[indexInArray + 1];
      gValue2 = gradImg[indexInArray];
      gValue3 = gradImg[indexInArray - 1];
      if (gValue1 > gValue2 && gValue1 > gValue3) {
        // Down-right
        px.x = px.x + 1;
      } else if (gValue3 > gValue2 && gValue3 > gValue1) {
        // Down-left
        px.x = px.x - 1;
      }

    } else if (lastY > px.y) {
      // ----------------
      // | Ok | Ok | Ok |
      // |  X |  X |  X |
      // |  X |  X |  X |
      // ----------------
      // Go up
      if (px.y == 0 || px.x == 0 || px.x == imageWidth - 1) {
        // Reach the image border
        return false;
      }

      indexInArray -= imageWidth;
      px.y--;
      gValue1 = gradImg[indexInArray + 1];
      gValue2 = gradImg[indexInArray];
      gValue3 = gradImg[indexInArray - 1];
      if (gValue1 > gValue2 && gValue1 > gValue3) {
        // Up-right
        px.x = px.x + 1;
      } else if (gValue3 > gValue2 && gValue3 > gValue1) {
        // Up-left
        px.x = px.x - 1;
      } // else Straight-up

    } else if (lastX < px.x) {  // lastY == px.y
      // ----------------
      // |  X |  X | Ok |
      // |  X |  X |  X |
      // |  X |  X | Ok |
      // ----------------
      if (px.x == imageWidth - 1 || px.y == 0 || px.y == imageHeight - 1) {
        // Reach the image border
        return false;
      }

      indexInArray++;
      px.x++;
      if (gradImg[indexInArray - imageWidth] > gradImg[indexInArray + imageWidth]) {
        // Up-right
        px.y--;
      } else {
        // Down-right
        px.y++;
      }

    } else {  // lastY == px.y && lastX > px.x
      // ----------------
      // | Ok |  X |  X |
      // |  X |  X |  X |
      // | Ok |  X |  X |
      // ----------------
      if (px.x == 0 || px.y == 0 || px.y == imageHeight - 1) {
        // Reach the image border
        return false;
      }

      indexInArray--;
      px.x--;
      if (gradImg[indexInArray - imageWidth] > gradImg[indexInArray + imageWidth]) {
        // Up-right
        px.y--;
      } else {
        // Down-right
        px.y++;
      }
    }
  }

  return gradImg[px.y * imageWidth + px.x];
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

bool EdgeDrawer::canSegmentBeExtended(FullSegmentInfo &segment,
                                      bool extendByTheEnd,
                                      ImageEdge &pixelsInTheExtension) {
  bool inlier, found, horizontalValidationDir, fitsEigenvaluesCond, fitsAngleCond;
  int i, validateDx, validateDy, xOffset, yOffset, indexInArray;
  double theta;
  float a, b, d, tmp, s1, s2, eigen_angle;
  Pixel firstPx, lastPx, px;
  ImageEdge pixelsToValidate;
  cv::Vec3d eq;
  firstPx = extendByTheEnd ? segment.getLastPixel() : segment.getFirstPixel();

  for (int GRAD_JUNCTION_SIZE: listOfJunctionSizes) {

    if (segment.getNumOfPixels() <= GRAD_JUNCTION_SIZE) break;

    // LOGD << "Trying to extend with step size = " << GRAD_JUNCTION_SIZE;

    px = firstPx;
    // calculate the pixel after the extension
    found = findNextPxWithProjection(gradImg,
                                     segment.getLineEquation(),
                                     !extendByTheEnd,
                                     imageWidth,
                                     imageHeight,
                                     GRAD_JUNCTION_SIZE,
                                     px);
    if (!found) {
      // LOGD << "Cannot follow pixel (" << px << "). Do not extending the segment.";
      continue;
    }
    // Get the pixels in the extension using the Bresenham Algorithm
    pixelsInTheExtension = upm::bresenham(firstPx.x, firstPx.y, px.x, px.y);
    lastPx = pixelsInTheExtension[pixelsInTheExtension.size() - 2];
    // LOGD << "\t\t\tLeaving free the edge drawing in px: " << px << ", lastPx: " << lastPx;

    // Try to draw GRAD_JUNCTION_SIZE pixels if the segment direction
    for (i = 0; i < GRAD_JUNCTION_SIZE; i++) {
      if (!findNextPxWithGradient(segment.horizontal() ? UPM_EDGE_HORIZONTAL : UPM_EDGE_VERTICAL,
                                  gradImg,
                                  imageWidth, imageHeight,
                                  px,
                                  lastPx)) {
        // LOGD << "\t\t\tNo new pixels found following the edge: CANNOT EXTEND";
        pixelsInTheExtension.clear();
        break;
      }

      // LOGD << "\t\t\t\tExtension pixel after the junction: " << px;
      inlier = segment.isInlier(px.x, px.y, pxToSegmentDistTh);
      if (inlier) {
        pixelsInTheExtension.push_back(px);
        pixelsToValidate.push_back(px);
      } else {
        // LOGD << "\t\t\tEdge Drawing is not able to follow the line";
        pixelsInTheExtension.clear();
        break;
      }
    }

    // If we couldn't draw the GRAD_JUNCTION_SIZE pixels in the line direction
    if (pixelsInTheExtension.empty()) continue;

      // Get the segment angle
      eq = segment.getLineEquation();
      // Get the angle perpendicular to the fitted line
      theta = std::atan2(eq[0], eq[1]) + M_PI_2;
      // Force theta to be in range [0, M_PI)
      while (theta < 0) theta += M_PI;
      while (theta >= M_PI) theta -= M_PI;

      // Evaluation of the extension pixels based on the auto-correlation gradient matrix
      // Elements of the matrix M = [[a, b], [b, d]]
      a = 0, b = 0, d = 0;
      validateDx = std::abs(pixelsToValidate.back().x - pixelsToValidate.front().x);
      validateDy = std::abs(pixelsToValidate.back().y - pixelsToValidate.front().y);
      horizontalValidationDir = validateDx >= validateDy;
      for (Pixel &extPixel: pixelsToValidate) {
        for (int offset: {-1, 0, 1}) {
          // Depending on the segment extension pixels orientation, look the vertical or horizontal neighbors
          xOffset = horizontalValidationDir ? 0 : offset;
          yOffset = horizontalValidationDir ? offset : 0;
          indexInArray = std::min(imageHeight-1, std::max(0, extPixel.y + yOffset)) * imageWidth
              + std::min(imageWidth-1, std::max(0, extPixel.x + xOffset));

          a += pDxImg[indexInArray] * pDxImg[indexInArray];
          b += pDxImg[indexInArray] * pDyImg[indexInArray];
          d += pDyImg[indexInArray] * pDyImg[indexInArray];
        }
      }
      // Manually compute SVD of matrix M = [[a, b], [b, d]]
      // https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
      tmp = a * a - d * d;
      s1 = a * a + 2 * b * b + d * d;
      s2 = std::sqrt(tmp * tmp + 4 * b * b * (a + d) * (a + d));
      eigen_angle = -0.5f * std::atan2(2 * a * b + 2 * b * d, tmp);
      while (eigen_angle < 0) eigen_angle += M_PI;
      while (eigen_angle >= M_PI) eigen_angle -= M_PI;

      // This conditions requires the most important eigenvalue to be significantly bigger than the second one
      fitsEigenvaluesCond = std::sqrt((s1 + s2) / (s1 - s2 + 0.00001)) > junctionEigenvalsTh;
      // This condition requires that the first eigenvector has a similar angle to the segment.
      fitsAngleCond = circularDist(theta, eigen_angle, M_PI) < junctionAngleTh;
      // LOGD << "Fits eigenvalues: " << fitsEigenvaluesCond << ", fits angle: " << fitsAngleCond;

      if (!fitsEigenvaluesCond || !fitsAngleCond) {
        continue;
      }

    // LOGD << "\t\t\tThe segment can be extended. Extension px's: " << pixelsInTheExtension;
    return true;

  }
  return false;
}

inline uint8_t EdgeDrawer::inverseDirection(uint8_t dir) {
  switch (dir) {
    case UPM_RIGHT:return UPM_LEFT;
    case UPM_LEFT: return UPM_RIGHT;
    case UPM_UP: return UPM_DOWN;
    default: return UPM_UP; // UPM_DOWN
  }
}

inline uint8_t EdgeDrawer::directionFromLineEq(const cv::Vec3d &eq) {
  double lineX = -eq[1];
  double lineY = eq[0];
  if (UPM_ABS(lineX) > UPM_ABS(lineY)) {
    // Horizontal line
    if (lineX > 0) return UPM_RIGHT;
    else return UPM_LEFT;
  } else {
    // Vertical line
    if (lineY > 0) return UPM_DOWN;
    else return UPM_UP;
  }
}

Segments EdgeDrawer::getDetectedSegments() const {
  Segments result(segments.size());
  for (int i = 0; i < segments.size(); i++) result[i] = segments[i].getEndpoints();
  return result;
}

bool EdgeDrawer::findNextPxWithProjection(const int16_t *gradImg,
                                          cv::Vec3f eq,
                                          bool invertLineDir,  // extendSecondEndpoint
                                          int imageWidth, int imageHeight,
                                          int stepSize,
                                          Pixel &px) {
  assert(px.x >= 0 && px.x < imageWidth && px.y >= 0 && px.y < imageHeight);

  int16_t gValue1, gValue2, gValue3, gValue4;
  cv::Point2f extendedPoint, lasPxReproj;

  lasPxReproj = getProjectionPtn(eq, cv::Point2f(px.x, px.y));

  if (!invertLineDir) {
    extendedPoint.x = lasPxReproj.x - stepSize * eq[1];
    extendedPoint.y = lasPxReproj.y + stepSize * eq[0];
  } else {
    extendedPoint.x = lasPxReproj.x + stepSize * eq[1];
    extendedPoint.y = lasPxReproj.y - stepSize * eq[0];
  }

  if (extendedPoint.x < 0 || extendedPoint.x >= imageWidth || extendedPoint.y < 0 || extendedPoint.y >= imageHeight) {
    return false;
  }
  int x = extendedPoint.x;
  int y = extendedPoint.y;
  gValue1 = gradImg[y * imageWidth + x];
  gValue2 = gradImg[y * imageWidth + std::min(x + 1, imageWidth - 1)];
  gValue3 = gradImg[std::min(y + 1, imageHeight - 1) * imageWidth + x];
  gValue4 = gradImg[std::min(y + 1, imageHeight - 1) * imageWidth + std::min(x + 1, imageWidth - 1)];

  // Select the bigger of the 4 values
  if (gValue2 > gValue1) {
    if (gValue3 > gValue2) {
      // gValue3 > gValue2 > gValue1
      if (gValue4 > gValue3) {
        //gValue4 is the bigger
        px.x = std::min(x + 1, imageWidth);
        px.y = std::min(y + 1, imageHeight);
        return gValue4;
      } else {
        //gValue3 is the bigger
        px.x = x;
        px.y = std::min(y + 1, imageHeight);
        return gValue3;
      }
    } else {
      // gValue2 >= (gValue1, gValue3)
      if (gValue4 > gValue2) {
        //gValue4 is the bigger
        px.x = std::min(x + 1, imageWidth);
        px.y = std::min(y + 1, imageHeight);
        return gValue4;
      } else {
        // gValue2 is the bigger
        px.x = std::min(x + 1, imageWidth);
        px.y = y;
        return gValue2;
      }
    }
  } else {
    // gValue1 >= gValue2
    if (gValue3 > gValue1) {
      // gValue3 > gValue1 >= gValue2
      if (gValue4 > gValue3) {
        //gValue4 is the bigger
        px.x = std::min(x + 1, imageWidth);
        px.y = std::min(y + 1, imageHeight);
        return gValue4;
      } else {
        //gValue3 is the bigger
        px.x = x;
        px.y = std::min(y + 1, imageHeight);
        return gValue3;
      }
    } else {
      // gValue1 >= (gValue2, gValue3)
      if (gValue4 > gValue1) {
        //gValue4 is the bigger
        px.x = std::min(x + 1, imageWidth);
        px.y = std::min(y + 1, imageHeight);
        return gValue4;
      } else {
        //gValue1 is the bigger
        px.x = x;
        px.y = y;
        return gValue1;
      }
    }
  }
}

}
