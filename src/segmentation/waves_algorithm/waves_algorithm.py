from collections import deque

import cv2
import numpy as np


def wave_algorithm(image, start_point, threshold=10):
    """
    Performs segmentation of an image region using a wave algorithm.

    :param image: Input grayscale image.
    :param start_point: Start point coordinate (x, y).
    :param threshold: Maximum intensity difference for a pixel belonging to a region.
    :return: Segmented area mask.
    """
    rows, cols = image.shape

    segmented_mask = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)

    start_intensity = image[start_point[1], start_point[0]]

    queue = deque([start_point])
    visited[start_point[1], start_point[0]] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()

        segmented_mask[y, x] = 255

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx]:
                if abs(int(image[ny, nx]) - int(start_intensity)) <= threshold:
                    queue.append((nx, ny))
                    visited[ny, nx] = True

    return segmented_mask


# image_path = '106_4_GaussianBlur.png'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# start_point = (191, 102)
#
# segmented_image = wave_algorithm(image, start_point, threshold=17)
#
# cv2.imwrite("106_4_waves.png", segmented_image)
#
# cv2.imwrite('segmented_result.png', segmented_image)
# cv2.imshow('Segmented Image', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
