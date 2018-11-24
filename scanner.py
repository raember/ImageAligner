# Created after guidance from:
# https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
import logging


class Transformer:
    log = None

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def create_smaller_copy(self, img, height=500):
        """
        Create a copy of the image to operate on for other steps.
        :param img: The image to copy
        :param height: The preferred height of the image
        :type height: int
        :return: A copy of the image and the ratio
        :rtype: object, float
        """
        copy = img.copy()
        ratio = float(copy.shape[0]) / height
        self.log.debug("Resizing image{} to a height of {}. Ratio: {}".format(
            copy.shape, height, ratio
        ))
        copy = imutils.resize(copy, height=height)
        self.log.debug("Done resizing{}.".format(copy.shape))
        return copy, ratio

    def detect_edges(self, img):
        """
        Detects the edges in an image.
        :param img: The image to scan for edges
        :return: images of the edges found
        """
        self.log.debug("Detecting edges.")
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        self.log.debug("Done detecting edges.")
        return edged

    def find_contours(self, img):
        """
        Finds the contours belonging to a document in an edge image.
        :param img: The edge image
        :return:
        """
        self.log.debug("Finding contours.")
        contours = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if imutils.is_cv2():
            contours = contours[0]
        else:
            contours = contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is None:
            self.log.error("Coundn't find contours in image.")
            raise Exception("No contours found.")
        # screenCnt = self._order_points(screenCnt)
        self.log.info("Found contours: {}".format(screenCnt).replace('\n\n', ','))
        return screenCnt

    def order_points(self, points):
        """
        Orders the four points.
        :param points: The points to order
        :return: ordered points
        """
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def warp_from_points(self, image, origin):
        (tl, tr, br, bl) = origin

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        destination = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        transform_matrix = cv2.getPerspectiveTransform(origin, destination)
        warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def to_grayscale(self, img):
        warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
        return warped
