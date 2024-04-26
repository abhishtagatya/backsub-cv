import argparse
import time
import os
import random

import cv2
import matplotlib.pyplot as plt

from backsub.util import load_image
from backsub.grabcut import GrabCutBSCV
from backsub.contourmask import ContourMaskBSCV
from backsub.watershedmask import WatershedMaskBSCV
from backsub.blend import color_balance, color_correction, gradient_mixing


if __name__ == '__main__':

    result = {
        'source': [],
        'target': [],
        'generated': [],
        'cb': [],
        'hms': [],
        'hmt': [],
        'gm': []
    }

    BASE_PICTURE = 'dataset/supervisely/human/images/'
    BASE_STYLE = 'dataset/resized/resized/'

    n = 4

    for i in range(n):
        r_image = (random.choice(os.listdir(BASE_PICTURE)))
        r_style = (random.choice(os.listdir(BASE_STYLE)))

        r_image_path = BASE_PICTURE + r_image
        r_style_path = BASE_STYLE + r_style

        src_image = load_image(r_image_path)
        tar_image = load_image(r_style_path)

        model = GrabCutBSCV(
            image=r_image_path,
            background_image=r_style_path,
            iterC=4
        )
        image, mask = model.process()

        cb_image = color_balance(image, percent=1)
        hms_image = color_correction(image, src_image)
        hmt_image = color_correction(image, tar_image)
        gm_image = gradient_mixing(src_image, tar_image, mask)

        result['source'].append(src_image)
        result['target'].append(tar_image)
        result['generated'].append(image)
        result['cb'].append(cb_image)
        result['hms'].append(hms_image)
        result['hmt'].append(hmt_image)
        result['gm'].append(gm_image)

    fig, axs = plt.subplots(n, 7, figsize=(5, 5))
    plt.subplots_adjust(wspace=0)

    # Remove grid lines
    for idx, ax in enumerate(axs):
        if idx == 0:
            ax[0].set_title('Source')
            ax[1].set_title('Target')
            ax[2].set_title('Generated')
            ax[3].set_title('Color Balance')
            ax[4].set_title('Hist. Match (Source)')
            ax[5].set_title('Hist. Match (Target)')
            ax[6].set_title('Gradient Mixing')

        ax[0].imshow(cv2.cvtColor(result['source'][idx], cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[1].imshow(cv2.cvtColor(result['target'][idx], cv2.COLOR_BGR2RGB))
        ax[1].axis('off')
        ax[2].imshow(cv2.cvtColor(result['generated'][idx], cv2.COLOR_BGR2RGB))
        ax[2].axis('off')
        ax[3].imshow(cv2.cvtColor(result['cb'][idx], cv2.COLOR_BGR2RGB))
        ax[3].axis('off')
        ax[4].imshow(cv2.cvtColor(result['hms'][idx], cv2.COLOR_BGR2RGB))
        ax[4].axis('off')
        ax[5].imshow(cv2.cvtColor(result['hmt'][idx], cv2.COLOR_BGR2RGB))
        ax[5].axis('off')
        ax[6].imshow(cv2.cvtColor(result['gm'][idx], cv2.COLOR_BGR2RGB))
        ax[6].axis('off')

    plt.tight_layout()
    plt.show()
