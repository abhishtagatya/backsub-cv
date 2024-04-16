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

        kernel = np.ones(self.kSize, np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.iterC)

        bg_area = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, fg_area = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        fg_area = np.uint8(fg_area)
        unknown = cv2.subtract(bg_area, fg_area)

        ret, markers = cv2.connectedComponents(fg_area)

        markers = markers + 1

        markers[unknown == 255] = 0

        markers = cv2.watershed(self.image, markers)

        fg_mask = np.where(markers == 1, 255, 0).astype('uint8')
        bg_mask = 255 - fg_mask

        background_filled_image = cv2.bitwise_and(self.background_image, self.background_image, mask=bg_mask)
        return (cv2.add(background_filled_image, cv2.bitwise_and(self.image, self.image, mask=fg_mask)),
                np.where((fg_mask == 0), 0, 255).astype('uint8'))

