import cv2
import numpy as np

from src.segmentation.preprocessing.cropping_black_area import crop_frame


def make_mask(marked_image_path):
    image = crop_frame(marked_image_path, 10)

    lower_red = np.array([0, 0, 255])
    upper_red = np.array([0, 0, 255])

    binary_mask = cv2.inRange(image, lower_red, upper_red)
    contours = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask


def make_inverted_mask(mask):
    return cv2.bitwise_not(mask)
