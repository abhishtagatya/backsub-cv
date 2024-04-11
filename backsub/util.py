import cv2


def load_image(f_path: str, resize=(720, 400)):
    image = cv2.imread(f_path)
    image = cv2.resize(image, resize)
    return image

