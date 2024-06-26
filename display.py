import argparse

import cv2
from backsub.grabcut import GrabCutBSCV
from backsub.contourmask import ContourMaskBSCV
from backsub.watershedmask import WatershedMaskBSCV
from backsub.util import load_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Method on Image")
    parser.add_argument("model", help=f"Model Type", choices=['contourmask', 'grabcut', 'watershed'])
    parser.add_argument("image_file", help=f"Path to Image File")
    parser.add_argument("background_file", help=f"Path to Substitute Background File")
    parser.add_argument("--file_out", help=f"Name to Export Result", default='result')
    args = parser.parse_args()

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
            iterC=4
        )

    if model is None:
        raise NotImplementedError(f'Unsupported model choice of `{args.model}`.')

    image, mask = model.process()

    cv2.imshow("Original", load_image(args.image_file))
    cv2.imshow("Background", load_image(args.background_file))
    cv2.imshow("Result", image)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

