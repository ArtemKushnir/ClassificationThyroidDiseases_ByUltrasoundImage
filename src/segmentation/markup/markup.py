from pathlib import Path

import cv2
import numpy as np


def make_markup_points(image_path, points):
    image = cv2.imread(image_path)

    for point in points:
        cv2.circle(image, (point["x"], point["y"]), radius=1, color=(0, 0, 255), thickness=-1)  # Красный цвет (BGR)

    return image


def make_markup_lines(image_path, points):
    image = cv2.imread(image_path)

    points_array = np.array([[point["x"], point["y"]] for point in points])

    points_array = points_array.reshape((-1, 1, 2))

    cv2.polylines(image, [points_array], isClosed=True, color=(0, 0, 255), thickness=1)

    return image


def write_marked_to_file(image_path, marked_image):
    image_name = Path(image_path).name
    marked_image_path = "../marked_image_box/marked_" + image_name

    cv2.imwrite(marked_image_path, marked_image)


image_path = "../image_box/108_1.png"

points = [
    {"x": 294, "y": 84},
    {"x": 287, "y": 84},
    {"x": 280, "y": 84},
    {"x": 269, "y": 84},
    {"x": 261, "y": 84},
    {"x": 255, "y": 84},
    {"x": 248, "y": 91},
    {"x": 244, "y": 100},
    {"x": 237, "y": 117},
    {"x": 237, "y": 123},
    {"x": 237, "y": 128},
    {"x": 240, "y": 135},
    {"x": 247, "y": 140},
    {"x": 254, "y": 145},
    {"x": 259, "y": 147},
    {"x": 266, "y": 149},
    {"x": 279, "y": 152},
    {"x": 293, "y": 152},
    {"x": 314, "y": 150},
    {"x": 322, "y": 150},
    {"x": 329, "y": 150},
    {"x": 346, "y": 138},
    {"x": 348, "y": 132},
    {"x": 348, "y": 126},
    {"x": 341, "y": 107},
    {"x": 335, "y": 102},
    {"x": 333, "y": 94},
    {"x": 332, "y": 89},
    {"x": 325, "y": 88},
    {"x": 319, "y": 86},
    {"x": 311, "y": 83},
    {"x": 303, "y": 82},
    {"x": 298, "y": 82},
    {"x": 291, "y": 82},
]

image = make_markup_lines(image_path, points)

write_marked_to_file(image_path, image)
