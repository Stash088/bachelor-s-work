import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import cv2
import utils

def calc_px_per_cm2(resolution):
    return int((resolution**2) / (2.54**2))

def find_leaves(binary_img, min_size):
    labels, _ = ndi.label(binary_img)
    items, area = np.unique(labels, return_counts=True)
    big_items = items[area > min_size][1:]  # subtract background
    leaf = np.isin(labels, big_items)  # keep items that are leaf
    item_areas = area[np.isin(items, big_items)]
    return leaf, item_areas

def main():
    # ...

    # Load image
    img = mpimg.imread('leaf.png')
    
    # Check if image has an alpha channel
    if img.shape[2] == 4:
        # Convert RGBA image to RGB
        img = img[:, :, :3]

    # Convert image to grayscale
    img_gray = rgb2gray(img)
    resolution = 300  # Resolution of the image in dots per inch
    min_size = 1000  # Minimum size of a leaf in pixels

    # Convert image to binary
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh
    binary = clear_border(binary)

    # Find leaves and calculate statistics
    leaf, item_areas = find_leaves(binary, min_size)
    px_per_cm2 = calc_px_per_cm2(resolution)
    num_leaves = len(item_areas)
    avg_area = np.mean(item_areas) / px_per_cm2
    total_area = np.sum(item_areas) / px_per_cm2

    # Print results
    print("Number of leaves:", num_leaves)
    print("Average leaf area:", avg_area, "cm^2")
    print("Total leaf area:", total_area, "cm^2")

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[1].imshow(leaf, cmap=plt.cm.binary)
    ax[1].set_title("Detected Leaves")
    plt.tight_layout()
    plt.show()


def fractal_dimension(image):
    mask = utils.mask_leaf(image)

    # Нахождение контуров объектов на изображении
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Вычисляем фрактальную размерность с помощью бокс-счета
    box_count = 0
    box_size = 2
    while box_size < image.shape[0]:
        for contour in contours:
            for point in contour:
                if point[0][0] % box_size == 0 and point[0][1] % box_size == 0:
                    box_count += 1
                    break
        box_size *= 2

    # Вычисляем фрактальную размерность
    fractal_dimension = np.log(box_count) / np.log(box_size)

    return fractal_dimension



# Пример использования
image = cv2.imread('data/leaf.png')
fractal_dimension = fractal_dimension(image)
print(f'Фрактальная размерность: {fractal_dimension}')