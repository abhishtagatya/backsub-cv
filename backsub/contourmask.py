import cv2
import numpy as np

from backsub import BaseBSCV


class ContourMaskBSCV(BaseBSCV):

    def __init__(self, image, background_image, resize=(720, 400), kSize=(5, 5)):
        super(ContourMaskBSCV, self).__init__(image, background_image, resize, kSize=kSize)

    def process(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # THRESH_BINARY : Binary Thresholding (0...255)
        # THRESH_OTSU   : Adaptive Thresholding based on Histogram (Minimize Intra-Class Variance)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 1. Scanning the Binary Image from L->R to Find Starting Point (Mark Location)
        # 2. From Starting Point, Check Neighboring Pixel for next Location (Repeat)
        # If no potential neighbor, stop search
        # 3. Continue until reach the starting point

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        # Fill in contours from marked locations

        kernel = np.ones(self.kSize, np.uint8)
        fg_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Fill in small gaps or holes in the mask by performing dilation followed by erosion,

        bg_mask = 255 - fg_mask
        # Invert mask

        background_filled_image = cv2.bitwise_and(self.background_image, self.background_image, mask=bg_mask)
        return (cv2.add(background_filled_image, cv2.bitwise_and(self.image, self.image, mask=fg_mask)),
                np.where((fg_mask == 0), 0, 255).astype('uint8'))
