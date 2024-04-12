import cv2


def calculate_iou(gt_mask, generated_mask):
    # Compute intersection (logical AND)
    intersection = cv2.bitwise_and(gt_mask, generated_mask)

    # Compute union (logical OR)
    union = cv2.bitwise_or(gt_mask, generated_mask)

    # Calculate areas of intersection and union
    area_intersection = cv2.countNonZero(intersection)
    area_union = cv2.countNonZero(union)

    # Calculate IoU
    iou = area_intersection / area_union if area_union > 0 else 0.0

    return iou


def calculate_pixel_accuracy(gt_mask, generated_mask):
    # Compute the number of pixels
    total_pixels = gt_mask.size

    # Compute the number of correctly classified pixels
    correct_pixels = cv2.countNonZero(cv2.bitwise_and(gt_mask, generated_mask))

    # Calculate pixel accuracy
    pixel_accuracy = correct_pixels / total_pixels

    return pixel_accuracy
