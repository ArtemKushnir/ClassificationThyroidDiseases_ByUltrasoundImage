import cv2

from src.segmentation.histograms.histograms import build_histogram
from src.segmentation.histograms.masks import make_inverted_mask, make_mask
from src.segmentation.preprocessing.cropping_black_area import crop_frame

marked_image_path = "../marked_image_box/marked_108_1.png"
image_path = "../image_box/108_1.png"

mask = make_mask(marked_image_path)
inverted_mask = make_inverted_mask(mask)

image = crop_frame(image_path, 10)

image = cv2.medianBlur(image, 3)

build_histogram(image)
build_histogram(image, mask)
build_histogram(image, inverted_mask, True)
