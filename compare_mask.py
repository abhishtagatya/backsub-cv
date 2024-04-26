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


if __name__ == '__main__':
    result = {
        'source': [],
        'mask': [],
        'cbms': [],
        'ws': [],
        'gc': []
    }

    BASE_PICTURE = 'dataset/supervisely/human/images/'
    BASE_MASK = 'dataset/supervisely/human/masks/'
    BASE_STYLE = 'dataset/resized/resized/'

    n = 6

    for i in range(n):
        r_image = (random.choice(os.listdir(BASE_PICTURE)))
        r_style = (random.choice(os.listdir(BASE_STYLE)))

        r_image_path = BASE_PICTURE + r_image
        r_mask_path = BASE_MASK + r_image
        r_style_path = BASE_STYLE + r_style

        src_image = load_image(r_image_path)
        msk_image = load_image(r_mask_path)
        tar_image = load_image(r_style_path)

        cm_model = ContourMaskBSCV(
            image=r_image_path,
            background_image=r_style_path,
            kSize=(11, 11)
        )
        _, cm_mask = cm_model.process()

        ws_model = WatershedMaskBSCV(
            image=r_image_path,
            background_image=r_style_path,
            kSize=(11, 11),
            iterC=3
        )
        _, ws_mask = ws_model.process()

        gc_model = GrabCutBSCV(
            image=r_image_path,
            background_image=r_style_path,
            iterC=5
        )
        _, gc_mask = gc_model.process()

        result['source'].append(src_image)
        result['mask'].append(msk_image)
        result['cbms'].append(cm_mask)
        result['ws'].append(ws_mask)
        result['gc'].append(gc_mask)

    fig, axs = plt.subplots(n, 5, figsize=(5, 5))
    plt.subplots_adjust(wspace=0)

    # Remove grid lines
    for idx, ax in enumerate(axs):
        if idx == 0:
            ax[0].set_title('Source')
            ax[1].set_title('Ground Truth')
            ax[2].set_title('Contour-based Morph.')
            ax[3].set_title('Watershed')
            ax[4].set_title('GrabCut')

        ax[0].imshow(cv2.cvtColor(result['source'][idx], cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[1].imshow(cv2.cvtColor(result['mask'][idx], cv2.COLOR_BGR2RGB), cmap='Greys')
        ax[1].axis('off')
        ax[2].imshow(cv2.cvtColor(result['cbms'][idx], cv2.COLOR_BGR2RGB), cmap='Greys')
        ax[2].axis('off')
        ax[3].imshow(cv2.cvtColor(result['ws'][idx], cv2.COLOR_BGR2RGB), cmap='Greys')
        ax[3].axis('off')
        ax[4].imshow(cv2.cvtColor(result['gc'][idx], cv2.COLOR_BGR2RGB), cmap='Greys')
        ax[4].axis('off')

    plt.tight_layout()
    plt.show()