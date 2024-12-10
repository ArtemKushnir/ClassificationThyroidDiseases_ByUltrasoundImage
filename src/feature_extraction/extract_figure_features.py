import json
import os
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops

from src.utils.get_paths import get_join_paths, get_paths_with_extension

FOLDER_CLEAN_IMAGE_PATH = (
    "/home/kush/machine_learning/ClassificationThyroidDiseases_ByUltrasoundImage/data/clear_and_cropped_image_data/"
)
FOLDER_IMAGE_PATH = "/home/kush/machine_learning/ClassificationThyroidDiseases_ByUltrasoundImage/data/image_data/"


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


def get_wavelet_features(image, mask):
    result = {}

    masked_image = image.copy()
    masked_image[mask != 255] = 0

    coeffs2 = pywt.dwt2(masked_image, "bior1.3")
    ll, (lh, hl, hh) = coeffs2

    wavelet_stats = {
        "LL_mean": np.mean(ll),
        "LH_mean": np.mean(lh),
        "HL_mean": np.mean(hl),
        "HH_mean": np.mean(hh),
        "LL_std": np.std(ll),
        "LH_std": np.std(lh),
        "HL_std": np.std(hl),
        "HH_std": np.std(hh),
        "LL_energy": np.sum(ll**2),
        "LH_energy": np.sum(lh**2),
        "HL_energy": np.sum(hl**2),
        "HH_energy": np.sum(hh**2),
    }
    for name, stat in wavelet_stats.items():
        result[f"figure_{name}"] = stat

    return result


def get_fourier_features(image, mask):
    result = {}

    masked_image = image.copy()
    masked_image[mask != 255] = 0

    f_transform = np.fft.fft2(masked_image)

    f_transform_shifted = np.fft.fftshift(f_transform)

    magnitude_spectrum = np.abs(f_transform_shifted)

    phase_spectrum = np.angle(f_transform_shifted)

    fourier_stats = {
        "mean_magnitude": np.mean(magnitude_spectrum),
        "std_magnitude": np.std(magnitude_spectrum),
        "max_magnitude": np.max(magnitude_spectrum),
        "min_magnitude": np.min(magnitude_spectrum),
        "mean_phase": np.mean(phase_spectrum),
        "std_phase": np.std(phase_spectrum),
    }

    for name, stat in fourier_stats.items():
        result[f"figure_{name}"] = stat

    return result


def process_json(json_data):
    result = []
    data = json.loads(json_data)
    for i in data:
        result.append(i["points"])
    return result


def get_glcm_matrix_features(mask, image):
    result = {}
    masked_image = image.copy()
    masked_image[mask != 255] = 0
    quantized_image = (masked_image / 16).astype(np.uint8)
    non_zero_pixels = quantized_image[mask == 255]
    processed_image = np.zeros_like(mask, dtype=np.uint8)
    processed_image[mask == 255] = non_zero_pixels

    processed_image = np.zeros_like(mask, dtype=np.uint8)
    processed_image[mask == 255] = non_zero_pixels

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(processed_image, distances, angles, levels=16, symmetric=True, normed=True)

    glcm_stats = {
        "glcm_contrast": graycoprops(glcm, "contrast"),
        "glcm_dissimilarity": graycoprops(glcm, "dissimilarity"),
        "glcm_homogeneity": graycoprops(glcm, "homogeneity"),
        "glcm_asm": graycoprops(glcm, "ASM"),
        "glcm_energy": np.sqrt(graycoprops(glcm, "ASM")),
        "glcm_correlation": graycoprops(glcm, "correlation"),
    }
    for name, stat in glcm_stats.items():
        result[f"figure_0_{name}"] = stat[0][0]
        result[f"figure_45_{name}"] = stat[0][1]
        result[f"figure_90_{name}"] = stat[0][2]
        result[f"figure_135_{name}"] = stat[0][3]

    return result


def get_feature(image, points_array):
    result = {}
    points = points_array[0]
    polygon = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)

    image_size = image.shape

    mask = np.zeros(image_size, dtype=np.uint8)
    cv.fillPoly(mask, [polygon], 255)

    result.update(get_glcm_matrix_features(mask, image))

    result.update(get_fourier_features(image, mask))

    result.update(get_wavelet_features(image, mask))

    area = cv.countNonZero(mask)
    intensity_values = image[mask == 255]
    mean_intensity = np.mean(intensity_values)
    max_intensity = np.max(intensity_values)
    min_intensity = np.min(intensity_values)
    std_intensity = np.std(intensity_values)
    perimeter = cv.arcLength(polygon, True)
    median = np.median(intensity_values)
    skewness = skew(intensity_values)
    kurtosis_value = kurtosis(intensity_values, fisher=True)

    quantile_steps = np.arange(0.05, 1.0, 0.05)
    quantiles = np.quantile(intensity_values, quantile_steps)
    for i, quantile in enumerate(quantiles):
        result[f"figure_{i + 1}_quantile"] = quantile

    result.update(
        {
            "figure_mean": mean_intensity,
            "figure_area": area,
            "figure_max": max_intensity,
            "figure_min": min_intensity,
            "figure_std": std_intensity,
            "figure_perimeter": perimeter,
            "figure_skewness": skewness,
            "figure_median": median,
            "figure_kurtosis": kurtosis_value,
        }
    )

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
