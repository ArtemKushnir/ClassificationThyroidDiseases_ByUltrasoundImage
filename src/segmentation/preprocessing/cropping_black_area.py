from pathlib import Path

import cv2


def crop_frame(image_path, cropping_radius):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(gray_image, 3)

    # Пороговая фильтрация для создания бинарного изображения (все, что не черное - белое) или можно исп Canny
    threshold_value, binary_image = cv2.threshold(denoised_image, 5, 255, cv2.THRESH_BINARY)

    # Поиск контуров EXTERNAL => только внешние контуры
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    max_contour = max(filtered_contours, key=cv2.contourArea)
    # max_index = max(range(len(filtered_contours)), key=lambda i: cv2.contourArea(contours[i]))

    # cv2.drawContours(image, max_contour, -1, (0, 255, 0), 1)  # BGR -> green, -1 -> все контуры

    # Получение ограничивающего прямоугольника для контура, w - weight, h - hight, х и у коорд верхнего левого угла
    x, y, w, h = cv2.boundingRect(max_contour)

    cropped_image = image[
        y + cropping_radius : y + h - cropping_radius, x + cropping_radius + 5 : x + w - cropping_radius
    ]

    return cropped_image


def write_cropped_to_file(image_path, cropped_image):
    image_name = Path(image_path).name
    cropped_image_path = "cropped_" + image_name

    cv2.imwrite(cropped_image_path, cropped_image)


# my_image = crop_frame("../image_box/157_1.png", 10)
# write_to_file("../image_box/157_1.png", my_image)
