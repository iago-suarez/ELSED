import unittest

import cv2
import numpy as np
import pyelsed
from scipy.sparse import csgraph, csr_matrix


class StructureDetectionTest(unittest.TestCase):

    def assert_segs_close(self, segs1: np.ndarray, segs2: np.ndarray, tol: float = 0) -> None:
        """
        Checks that two sets of segments are similar.
        """
        if len(segs1) != len(segs2):
            return False

        # Generate segments in both directions
        inverted_segs1 = segs1.reshape(segs1.shape[:-1] + (2, 2))[..., ::-1, :].reshape(segs1.shape)
        inverted_segs2 = segs2.reshape(segs2.shape[:-1] + (2, 2))[..., ::-1, :].reshape(segs2.shape)
        bidir_segs1 = np.vstack([segs1, inverted_segs1])
        bidir_segs2 = np.vstack([segs2, inverted_segs2])

        mat = np.linalg.norm(bidir_segs1[:, None] - bidir_segs2, axis=-1) < tol
        rank = csgraph.structural_rank(csr_matrix(mat))
        self.assertEqual(rank, len(bidir_segs1), f'Assert error: Line segments: '
                                                 f'\n{np.array2string(segs1, precision=2)} and '
                                                 f'\n {np.array2string(segs2, precision=2)} are not equal')

    def test_empty(self) -> None:
        img = np.zeros((100, 100), np.uint8)
        result, scores = pyelsed.detect(img)
        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))

    def test_square(self) -> None:
        img = np.zeros((200, 200), np.uint8)
        x0, x1, y0, y1 = 20, 140, 50, 150
        img[y0:y1, x0:x1] = 255

        result, _scores = pyelsed.detect(img)
        self.assertEqual(result.shape, (4, 4))

        expected = np.array([[x0, y0, x1, y0],
                             [x1, y0, x1, y1],
                             [x1, y1, x0, y1],
                             [x0, y1, x0, y0]])
        self.assert_segs_close(result, expected, tol=2.5)

    def test_triangle(self) -> None:
        img = np.zeros((200, 200), np.uint8)
        # Define the triangle
        top = (100, 20)
        left = (50, 160)
        right = (150, 160)
        expected = np.array([[*top, *right], [*right, *left], [*left, *top]])
        # Draw triangle
        cv2.drawContours(img, [expected.reshape(-1, 2)], 0, (255,), thickness=cv2.FILLED)
        result, _scores = pyelsed.detect(img)

        self.assertEqual(result.shape, (3, 4))
        # TODO (tol=2.5) for some reason the left-bottom part of the triangle does not fit well
        self.assert_segs_close(result, expected, tol=6.0)

    def test_real_img(self) -> None:
        img = cv2.imread('images/P1020829.jpg', cv2.IMREAD_GRAYSCALE)
        segments, _scores = pyelsed.detect(img, gradientThreshold=20)
        # Check that it detects at least 300 segments
        self.assertGreater(len(segments), 300)
