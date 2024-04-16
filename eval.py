import argparse
import os

import cv2
from backsub.grabcut import GrabCutBSCV
from backsub.contourmask import ContourMaskBSCV
from backsub.watershedmask import WatershedMaskBSCV
from backsub.util import load_image

from backsub.metric import calculate_iou
from backsub.metric import calculate_pixel_accuracy

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Foreground Segmentation Methods")
    parser.add_argument("model", help=f"Model Type", choices=['contourmask', 'grabcut', 'watershed'])
    parser.add_argument("image_path", help=f"Path to Images")
    parser.add_argument("mask_path", help=f"Path to Image Masks")
    parser.add_argument("background", help=f"Path to Substitute Background")
    args = parser.parse_args()

    iou_list = []
    acc_list = []

    model = None
    for img in tqdm(os.listdir(args.image_path)):

        img_path = args.image_path + str(img)
        gt_mask = load_image(args.mask_path + str(img))
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

        if args.model == 'contourmask':
            model = ContourMaskBSCV(
                image=img_path,
                background_image=args.background,
                kSize=(11, 11)
            )

        if args.model == 'watershed':
            model = WatershedMaskBSCV(
                image=img_path,
                background_image=args.background,
                kSize=(11, 11),
                iterC=3
            )

        if args.model == 'grabcut':
            model = GrabCutBSCV(
                image=img_path,
                background_image=args.background,
                iterC=4
            )

        if model is None:
            raise NotImplementedError(f'Unsupported model choice of `{args.model}`.')

        _, mask = model.process()
        iou_list.append(calculate_iou(gt_mask, mask))
        acc_list.append(calculate_pixel_accuracy(gt_mask, mask))

    print(f"Evaluation of Model : {model}")
    print(f"Avg. IoU : {sum(iou_list) / len(iou_list):.4f}")
    print(f"Min. IoU : {min(iou_list):.4f}")
    print(f"Max. IoU : {max(iou_list):.4f}")

    print(f"Avg. PA  : {sum(acc_list) / len(acc_list):.4f}")
    print(f"Min. PA  : {min(acc_list):.4f}")
    print(f"Max. PA  : {max(acc_list):.4f}")

