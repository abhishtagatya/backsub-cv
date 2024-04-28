import cv2
import numpy as np

from backsub.util import load_image

from backsub import BaseBSCV


class WatershedMaskBSCV(BaseBSCV):

    def __init__(self, image, background_image, resize=(720, 400), kSize=(5, 5), iterC=2):
        super(WatershedMaskBSCV, self).__init__(image, background_image, resize, kSize=kSize, iterC=iterC)

    def process(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Obtain a binary image to help separate the foreground from the background.

        kernel = np.ones(self.kSize, np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.iterC)
        # Noise removal using morphological opening

        bg_area = cv2.dilate(opening, kernel, iterations=3)
        # Expand the Background Area

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, fg_area = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Assigns each pixel a value corresponding to its distance from the nearest zero pixel (Background)
        # Then apply threshold to the distance transform to obtain the foreground area

        fg_area = np.uint8(fg_area)
        unknown = cv2.subtract(bg_area, fg_area)
        # Any area that is nor background nor foreground is marked as unkown

        ret, markers = cv2.connectedComponents(fg_area)
        # Label the foreground area of the image, where each connected component corresponds to 
        # a distinct object or region of interest

        markers = markers + 1

        markers[unknown == 255] = 0
        # Marker values are adjusted to ensure that the background is not labeled as 0 and
        # unknown values are 0

        markers = cv2.watershed(self.image, markers)
        # Apply the watershed algorithm on the marker to segment the image

        fg_mask = np.where(markers == 1, 255, 0).astype('uint8')
        bg_mask = 255 - fg_mask
        # Adjust the mask values to 0 - 255 ranges

        background_filled_image = cv2.bitwise_and(self.background_image, self.background_image, mask=bg_mask)
        return (cv2.add(background_filled_image, cv2.bitwise_and(self.image, self.image, mask=fg_mask)),
                np.where((fg_mask == 0), 0, 255).astype('uint8'))

