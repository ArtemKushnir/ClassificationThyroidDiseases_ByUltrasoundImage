import json
import os
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np

from src.utils.get_paths import get_join_paths, get_paths_with_extension

FOLDER_CLEAN_IMAGE_PATH = "../data/clear_and_cropped_image_data/"
FOLDER_IMAGE_PATH = "../data/image_data/"


def get_points_from_images(folder_image_path=FOLDER_CLEAN_IMAGE_PATH, folder_xml_path=FOLDER_IMAGE_PATH):
    figures = {}

    all_image_paths = os.listdir(folder_image_path + "normal/") + os.listdir(folder_image_path + "malignant/")

    xml_paths = get_paths_with_extension(folder_xml_path, "xml")
    full_paths = get_join_paths(folder_xml_path, xml_paths)

    for path in all_image_paths:
        image_number = path[path.find("_") + 1 : path.rfind(".")]
        target_xml_number = image_number[:-2]

        target_xml = None
        for i in full_paths:
            xml_number = i[i.rfind("/") + 1 : i.rfind(".")]
            if target_xml_number == xml_number:
                target_xml = i

        root = ET.parse(target_xml)
        results = {}
        for mark in root.findall("mark"):
            count_image = mark.find("image").text
            svg_content = mark.find("svg").text
            svg = process_json(svg_content)
            results[target_xml_number + "_" + count_image] = svg
        figures[image_number] = results[image_number]
    return figures


def process_json(json_data):
    result = []
    data = json.loads(json_data)
    for i in data:
        result.append(i["points"])
    return result


def get_feature(image, points_array):
    result = {}
    for points in points_array:
        polygon = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)

        image_size = image.shape

        mask = np.zeros(image_size, dtype=np.uint8)
        cv.fillPoly(mask, [polygon], 255)

        area = cv.countNonZero(mask)
        intensity_values = image[mask == 255]
        mean_intensity = np.mean(intensity_values)
        max_intensity = np.max(intensity_values)
        min_intensity = np.min(intensity_values)
        std_intensity = np.std(intensity_values)
        perimeter = cv.arcLength(polygon, True)

        for key, value in {
            "figure_mean": mean_intensity,
            "figure_area": area,
            "figure_max": max_intensity,
            "figure_min": min_intensity,
            "figure_std": std_intensity,
            "figure_perimeter": perimeter,
        }.items():
            if key in result:
                result[key] = (result[key] + value) / 2
            else:
                result[key] = np.float64(value)
    return result


def extract_figure_features(folder_image_path=FOLDER_IMAGE_PATH):
    figures = get_points_from_images()
    image_paths = get_paths_with_extension(folder_image_path, "jpg")
    full_paths = get_join_paths(folder_image_path, image_paths)
    features = {}

    for i, image_path in enumerate(full_paths):
        image_number = image_paths[i][:-4]
        if image_number in figures:
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            features[image_number] = get_feature(image, figures[image_number])
    return features
