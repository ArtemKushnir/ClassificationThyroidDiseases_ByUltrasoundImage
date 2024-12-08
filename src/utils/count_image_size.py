# import cv2 as cv
# from get_paths import get_paths_with_extension, get_join_paths
#
# folder_path = "../data/cropped_image_data/"
# image_paths = get_paths_with_extension(folder_path, "jpg")
# full_image_paths = get_join_paths(folder_path, image_paths)
#
# image_size = {}
#
# for image_path in full_image_paths:
#     image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     image_size[image.shape] = image_size.get(image.shape, 0) + 1
#
# print(image_size)
