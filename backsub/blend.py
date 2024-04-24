import cv2
import numpy as np


def color_balance(image, percent=1):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)

    # Compute the histogram of the L channel
    hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])

    # Compute the cumulative distribution function of the L channel
    cdf_l = hist_l.cumsum() / hist_l.sum()

    # Determine the lower and upper bounds for the L channel
    low_bound = np.percentile(l, percent / 2)
    high_bound = np.percentile(l, 100 - percent / 2)

    # Clip the L channel to the specified bounds
    clipped_l = np.clip(l, low_bound, high_bound)

    # Scale the clipped L channel to the original range
    scaled_l = np.uint8(255 * (clipped_l - clipped_l.min()) / (clipped_l.max() - clipped_l.min()))

    # Merge the balanced L channel with the original A and B channels
    balanced_lab_image = cv2.merge((scaled_l, a, b))

    # Convert the balanced LAB image back to BGR color space
    balanced_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2BGR)

    return balanced_image


def color_correction(source, target):
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Split channels
    source_l, source_a, source_b = cv2.split(source_lab)
    target_l, target_a, target_b = cv2.split(target_lab)

    # Perform histogram matching for each channel
    corrected_l = match_histograms(source_l, target_l)
    corrected_a = match_histograms(source_a, target_a)
    corrected_b = match_histograms(source_b, target_b)

    # Merge channels
    corrected_lab = cv2.merge((corrected_l, corrected_a, corrected_b))

    # Convert back to BGR color space
    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    return corrected


def match_histograms(source, target):
    # Calculate histograms
    source_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256])

    # Calculate cumulative distribution functions (CDFs)
    source_cdf = source_hist.cumsum()
    target_cdf = target_hist.cumsum()

    # Normalize CDFs
    source_cdf_norm = source_cdf / source_cdf[-1]
    target_cdf_norm = target_cdf / target_cdf[-1]

    # Compute the mapping
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(source_cdf_norm - target_cdf_norm[i]))

    # Apply the mapping
    matched = mapping[source]

    return matched.astype('uint8')


def gradient_mixing(image1, image2, gradient_mask, iterC=3, dimF=0.3):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    # Perform iterative dilation with decreasing intensity
    dimmed_mask = gradient_mask.copy()
    for i in range(iterC):
        # Dilate the mask
        dilated_mask = cv2.dilate(dimmed_mask, kernel, iterations=1)
        # Dim the mask by multiplying with the dimming factor
        dilated_mask = cv2.multiply(dilated_mask, dimF)
        # Apply the dilated mask
        dimmed_mask = cv2.bitwise_or(dimmed_mask, dilated_mask)
        print(np.unique(dimmed_mask))

    # Normalize the gradient mask to range [0, 1]
    gradient_mask = dimmed_mask / 255.0

    # Invert the gradient mask
    inverted_mask = 1 - gradient_mask

    # Blend each channel using the gradient mask
    blended_image = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))

    blended_image[:, :, 0] = cv2.multiply(gradient_mask, image1[:, :, 0]) + cv2.multiply(inverted_mask, image2[:, :, 0])
    blended_image[:, :, 1] = cv2.multiply(gradient_mask, image1[:, :, 1]) + cv2.multiply(inverted_mask, image2[:, :, 1])
    blended_image[:, :, 2] = cv2.multiply(gradient_mask, image1[:, :, 2]) + cv2.multiply(inverted_mask, image2[:, :, 2])

    blended_image = blended_image.astype(np.uint8)

    return blended_image
