import cv2
import numpy as np
from skimage.color import label2rgb
from skimage.segmentation import slic


def calculate_params(region):
    mean = np.mean(region)
    variance = np.var(region)
    std_dev = np.std(region)
    return mean, variance, std_dev


def draw_points_on_image(image, points, color=(0, 255, 0), radius=3):
    output_image = image.copy()

    for point in points:
        cv2.circle(output_image, point, radius, color, -1)

    return output_image


def get_points_by_slic(image, points_number, segments_number=100, compactness=10, blur=3):
    segments = slic(image, n_segments=segments_number, compactness=compactness, sigma=blur)

    segmented_image = label2rgb(segments, image, kind="avg")
    cv2.imwrite("slic.png", segmented_image)

    global_mean, global_variance, global_std_dev = calculate_params(image)
    global_params = np.array([global_mean, global_variance, global_std_dev])

    deviations = []

    for cluster_number in np.unique(segments):
        mask = segments == cluster_number
        region = image[mask]

        local_mean, local_variance, local_std_dev = calculate_params(region)
        local_params = np.array([local_mean, local_variance, local_std_dev])

        cluster_points = np.argwhere(segments == cluster_number)
        centroid = tuple(np.mean(cluster_points, axis=0).astype(int))

        deviation = np.linalg.norm(local_params - global_params)
        deviations.append(((centroid[1], centroid[0]), deviation))

    deviations.sort(key=lambda x: x[1], reverse=True)

    results_points = []
    for i, (coordinates, deviation) in enumerate(deviations[:points_number]):
        results_points.append(coordinates)

    return results_points


# image = cv2.imread("cropped_106_4.png")
# if image is None:
#     raise FileNotFoundError("No such file.")
#
# #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# points = get_points_by_slic(image, 10, 80)
#
# new_image = draw_points_on_image(image, points)
# cv2.imwrite("result_slic.png", new_image)
