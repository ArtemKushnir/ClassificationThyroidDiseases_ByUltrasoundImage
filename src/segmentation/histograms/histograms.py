import cv2
import matplotlib.pyplot as plt


def build_histogram(image, mask=None, is_inverted=None):
    if mask is not None and not is_inverted:
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        plt.plot(hist)
        plt.title("Pixel distribution of selected area")
    elif mask is not None and is_inverted:
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        plt.plot(hist)
        plt.title("Pixel distribution of background")
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.title("Pixel distribution of all image")

    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()
