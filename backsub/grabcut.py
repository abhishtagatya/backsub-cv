import cv2
import numpy as np

from backsub import BaseBSCV


class GrabCutBSCV(BaseBSCV):

    def __init__(self, image, background_image, resize=(720, 400), iterC=3):
        super(GrabCutBSCV, self).__init__(image, background_image, resize, iterC=iterC)
        self.rect = (16, 16, *self.resize)

    def process(self):
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

