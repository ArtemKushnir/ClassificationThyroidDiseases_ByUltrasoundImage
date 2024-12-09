import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from src.segmentation.preprocessing.cropping_black_area import crop_frame

image_path = "../image_box/106_4.png"

image = crop_frame(image_path, 10)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# smoothed_image = cv2.medianBlur(image, 3)
# cv2.imwrite("106_4_medianBlur.png", smoothed_image)
#
# contrast_image = cv2.equalizeHist(smoothed_image)
# cv2.imwrite("106_4_medianBlur_hist.png", contrast_image)

smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imwrite("106_4_GaussianBlur.png", smoothed_image)

# equalized_image = cv2.equalizeHist(smoothed_image)
# cv2.imwrite("106_4_GaussianBlur_hist.png", equalized_image)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
contrast_image = clahe.apply(smoothed_image)
cv2.imwrite("106_4_clahe.png", contrast_image)

pixels = contrast_image.reshape(-1, 1)  # преобразование в одномерный массив (вектор), -1 -> кол-во строк автоматически

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixels)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

segmented_pixels = centers[labels]
segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Original image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Preprocessing")
plt.imshow(contrast_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Segmented image ({n_clusters} clusters)")
plt.imshow(segmented_image, cmap="gray")
plt.axis("off")

plt.show()

# plt.figure(figsize=(15, 5))
#
# plt.subplot(1, 3, 1)
# plt.title("Original image")
# plt.imshow(image, cmap='gray')
# plt.axis("off")
#
# plt.subplot(1, 3, 2)
# plt.title("Image with GaussianBlur")
# plt.imshow(smoothed_image_2, cmap='gray')
# plt.axis("off")
#
# plt.subplot(1, 3, 3)
# plt.title("Image with equalizeHist and GaussianBlur")
# plt.imshow(equalized_image, cmap='gray')
# plt.axis("off")
#
# plt.show()
