import cv2
import numpy as np

from backsub.util import load_image


class GrabCutBSCV:

    MODE_MULTIPLY = "multiply"
    MODE_BITAND = "bitand"

    def __init__(self, image, background_image, resize=(720, 400), iterC=3):
        self.image = load_image(image, resize)
        self.background_image = load_image(background_image, resize)
        self.rect = (16, 16, *resize)
        self.iterC = iterC

    def process(self, mode=MODE_BITAND):
        mask = np.zeros(self.image.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            self.image,
            mask=mask,
            rect=self.rect,
            bgdModel=bgdModel,
            fgdModel=fgdModel,
            iterCount=self.iterC,
            mode=cv2.GC_INIT_WITH_RECT
        )

        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        bg_mask = 255 - fg_mask

        background_filled_image = cv2.bitwise_and(self.background_image, self.background_image, mask=bg_mask)
        return (cv2.add(background_filled_image, cv2.bitwise_and(self.image, self.image, mask=fg_mask)),
                np.where((fg_mask == 0), 0, 255).astype('uint8'))

