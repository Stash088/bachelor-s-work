import cv2
import numpy as np

def mask_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([15, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_green, mask_brown)
    mask = cv2.bitwise_or(mask, mask_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def venation_leaf(image):
    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    venation = ""
    for ellipse in ellipses:
        angle = ellipse[2]
        if 45 <= angle <= 135:
            venation = "Параллельная"
        elif -45 >= angle >= -135:
            venation = "Перпендикулярная"
        else:
            venation = "Другая венация"
    return venation



def classfier_leaf(image):
    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)
    leaf_shape = ""
    for contour in approx_contours:
        vertices = len(contour)
    if vertices == 3:
        leaf_shape = "Треугольная"
    elif vertices == 4:
        leaf_shape = "Прямоугольная или квадратная"
    elif vertices == 5:
        leaf_shape = "Пятиугольная"
    else:
        leaf_shape = "Другая форма"
    return leaf_shape

def measure_extract(image):
    if isinstance(image, str):
        try:
            image = cv2.imread(image)
        except Exception as e:
            print("Error loading image:", str(e))
            return None, None, None, None

    if image is None:
        print("Invalid image")
        return None, None, None, None

    mask = mask_leaf(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No leaf contours found")
        return None, None, None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    x, y, w, h = cv2.boundingRect(largest_contour)
    top = (x + w // 2, y)
    bottom = (x + w // 2, y + h - 1)

    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    contour_area = cv2.contourArea(largest_contour)
    length = abs(bottom[1] - top[1])
    width = abs(rightmost[0] - leftmost[0])
    conversion_factor = 37.936267
    scaled_length = length / conversion_factor
    scaled_width = width / conversion_factor
    scaled_area = contour_area / (conversion_factor ** 2)


    center_x = int(np.mean(largest_contour[:, :, 0]))
    center_y = int(np.mean(largest_contour[:, :, 1]))

    distances_x = []
    distances_y = []
    for point in largest_contour:
        distance_x = point[0][0] - center_x
        distance_y = point[0][1] - center_y
        distances_x.append(distance_x)
        distances_y.append(distance_y)

    abs_diff_x = np.abs(distances_x)
    abs_diff_y = np.abs(distances_y)

    mean_abs_diff_x = np.mean(abs_diff_x)
    mean_abs_diff_y = np.mean(abs_diff_y)

    fluctuating_asymmetry = (mean_abs_diff_x / mean_abs_diff_y) if mean_abs_diff_y != 0 else 1

    cv2.drawContours(image, [largest_contour], -1, (255, 0, 0), thickness=3)
    cv2.circle(image, top, 9, (0, 0, 255), -1)
    cv2.circle(image, bottom, 9, (0, 0, 255), -1)
    cv2.circle(image, leftmost, 9, (0, 255, 0), -1)
    cv2.circle(image, rightmost, 9, (0, 255, 0), -1)

    return scaled_length,scaled_width,scaled_area , fluctuating_asymmetry

def local_enhancement(image_path):

    img = cv2.imread(image_path,0)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    img_equalized = clahe.apply(img)

    h, w = img.shape
    img_GB = cv2.bilateralFilter(img, 3, 50, 50)
    canny_threshold_common = [40, 100]
    canny_threshold_enhanced = [30, 60]
    edge_canny_up = cv2.Canny(img_GB[:round(h/6), round(w/2)-20:round(w/2)+20], *canny_threshold_common, apertureSize=3)
    edge_canny_middle = cv2.Canny(img_GB[round(h/6):round(h/2), round(w/2)-20:round(w/2)+20],
                                  *canny_threshold_enhanced, apertureSize=3)
    t = list(edge_canny_middle.ravel() == 255).count(1)/len(list(edge_canny_middle.ravel()))

    while not 1/40 < t < 1/20:
        # print(t)
        if t <= 1/40:
            canny_threshold_enhanced = [canny_threshold_enhanced[0] - 1, canny_threshold_enhanced[1] - 1]
        else:
            canny_threshold_enhanced = [canny_threshold_enhanced[0] + 1, canny_threshold_enhanced[1] + 1]
        edge_canny_middle = cv2.Canny(img_GB[round(h / 6):round(h / 2), round(w / 2) - 20:round(w / 2) + 20],
                                      *canny_threshold_enhanced, apertureSize=3)
        t = list(edge_canny_middle.ravel() == 255).count(1) / len(list(edge_canny_middle.ravel()))
    edge_canny_down = cv2.Canny(img_GB[round(h/2):, round(w/2)-20:round(w/2)+20], *canny_threshold_common, apertureSize=3)
    edge_canny_middle_horizontally = np.vstack((edge_canny_up, edge_canny_middle, edge_canny_down))
    edge_canny_left = cv2.Canny(img_GB[:, :round(w/2)-20], *canny_threshold_common, apertureSize=3)
    edge_canny_right = cv2.Canny(img_GB[:, round(w/2)+20:], *canny_threshold_common, apertureSize=3)
    edge_canny = np.hstack((edge_canny_left, edge_canny_middle_horizontally, edge_canny_right))

    edge_equalized = cv2.Canny(img_equalized, *canny_threshold_common, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    edge_canny = cv2.dilate(edge_canny, kernel)
    edge_canny = cv2.dilate(edge_canny, kernel)
    edge_canny = cv2.dilate(edge_canny, kernel)
    edge_canny = cv2.erode(edge_canny, kernel)

    edge_canny = cv2.morphologyEx(edge_canny, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_canny = cv2.morphologyEx(edge_canny, cv2.MORPH_CLOSE, kernel2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edge_equalized = cv2.dilate(edge_equalized, kernel)
    edge_equalized = cv2.dilate(edge_equalized, kernel)
    edge_equalized = cv2.dilate(edge_equalized, kernel)
    edge_equalized = cv2.erode(edge_equalized, kernel)

    edge_equalized = cv2.morphologyEx(edge_equalized, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_equalized = cv2.morphologyEx(edge_equalized, cv2.MORPH_CLOSE, kernel2)

    return img, img_equalized, edge_canny, edge_equalized