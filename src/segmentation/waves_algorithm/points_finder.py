import cv2
import numpy as np


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


def get_differing_points(image, points_number, window_size=80, step=40):
    """
    Search for points with the greatest deviation characteristics.

    :param image: Grayscale input image.
    :param points_number: Number of points.
    :param window_size: Sliding window size.
    :param step: Step for sliding window.
    :return: Points with the greatest deviations.
    """
    global_mean, global_variance, global_std_dev = calculate_params(image)

    global_params = np.array([global_mean, global_variance, global_std_dev])

    height, width = image.shape

    deviations = []

    for y in range(0, height - window_size + 1, step):
        for x in range(0, width - window_size + 1, step):
            window = image[y : y + window_size, x : x + window_size]

            local_mean, local_variance, local_std_dev = calculate_params(window)

            local_params = np.array([local_mean, local_variance, local_std_dev])

            deviation = np.linalg.norm(local_params - global_params)
            deviations.append(((x, y), deviation))

    deviations.sort(key=lambda x: x[1], reverse=True)

    results_points = []
    for i, (coordinates, deviation) in enumerate(deviations[:points_number]):
        results_points.append(coordinates)

    return results_points


# image = cv2.imread("cropped_108_1.png")
# if image is None:
#     raise FileNotFoundError("Не удалось загрузить изображение. Проверьте путь к файлу.")
#
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# points = get_differing_points(gray_image, 20, 80, 40)
#
# new_image = draw_points_on_image(image, points)
# cv2.imwrite("result.png", new_image)
