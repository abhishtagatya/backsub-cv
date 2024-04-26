import argparse
import time
import os

import cv2

from backsub.util import load_image
from backsub.grabcut import GrabCutBSCV
from backsub.contourmask import ContourMaskBSCV
from backsub.watershedmask import WatershedMaskBSCV
from backsub.blend import color_balance, color_correction, gradient_mixing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Method on Image")
    parser.add_argument("model", help=f"Model Type", choices=['contourmask', 'grabcut', 'watershed'])
    parser.add_argument("image_file", help=f"Path to Image File")
    parser.add_argument("background_file", help=f"Path to Substitute Background File")
    parser.add_argument("--blend_mode", help=f"Blending Mode (gradient_mix, balance, hist_match_s, hist_match_t, none)",
                        choices=(
                            "gradient_mix", "balance", "hist_match_s", "hist_match_t", "none",
                        ), default='none'
    )
    parser.add_argument("--file_out", help=f"Name to Export Result", default='result')
    args = parser.parse_args()

    src = load_image(args.image_file)
    tar = load_image(args.background_file)

    model = None
    if args.model == 'contourmask':
        model = ContourMaskBSCV(
            image=args.image_file,
            background_image=args.background_file,
            kSize=(11, 11)
        )

    if args.model == 'watershed':
        model = WatershedMaskBSCV(
            image=args.image_file,
            background_image=args.background_file,
            kSize=(11, 11),
            iterC=3
        )

    if args.model == 'grabcut':
        model = GrabCutBSCV(
            image=args.image_file,
            background_image=args.background_file,
            iterC=3
        )

    if model is None:
        raise NotImplementedError(f'Unsupported model choice of `{args.model}`.')

    image, mask = model.process()

    if args.blend_mode == "gradient_mix":
        image = gradient_mixing(src, tar, mask)

    if args.blend_mode == "hist_match_s":
        image = color_correction(image, src)

    if args.blend_mode == "hist_match_t":
        image = color_correction(image, tar)

    if args.blend_mode == "balance":
        image = color_balance(image, percent=1)

    cv2.imwrite(f"{args.file_out}.png", image)
