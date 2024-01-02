import cv2
import numpy as np
from skimage import measure, color


def get_initial_seed(img, side):
    """
    function: 获得增长初始点
    :param img: 源二值边缘
    :param side: 增长方向
    :return:initial_seed: 初始种子点, direction: 增长掩码
    """
    seed = []
    h, w = img.shape
    flag = 0
    initial_seed, direction = (0, 0), 0
    if side == 'top':
        direction = [[1, 1], [0, 1], [-1, 1], [1, 0], [-1, 0]]
        for i in range(h):
            for j in range(round(w/2)-20, round(w/2)+20):
                if img[i, j] == 255:
                    seed.append((j, i))
                    flag = 1
            if flag:
                initial_seed = seed[int(len(seed) / 2)]
                break
    elif side == 'bottom':
        direction = [[-1, -1], [0, -1], [1, -1], [1, 0], [-1, 0]]
        for i in range(h - 1, -1, -1):
            for j in range(round(w/2)-20, round(w/2)+20):
                if img[i, j] == 255:
                    seed.append((j, i))
                    flag = 1
            if flag:
                initial_seed = seed[int(len(seed) / 2)]
                break
    elif side == 'left':
        direction = [[0, -1], [1, -1], [1, 1], [0, 1], [1, 0]]
        for j in range(round(w/2)-20, round(w/2)+20):
            for i in range(h):
                if img[i, j] == 255:
                    seed.append((j, i))
                    flag = 1
            if flag:
                initial_seed = seed[int(len(seed) / 2)]
                break
    elif side == 'right':
        direction = [[-1, -1], [0, -1], [0, 1], [-1, 1], [-1, 0]]
        for j in range(round(w/2)+20-1, round(w/2)-20-1, -1):
            for i in range(h):
                if img[i, j] == 255:
                    seed.append((j, i))
                    flag = 1
            if flag:
                initial_seed = seed[int(len(seed) / 2)]
                break
    elif side == 'all':
        direction = [[-1, -1], [0, -1], [1, -1], [1, 1], [0, 1], [-1, 1], [1, 0], [-1, 0]]
        for i in range(h - 1, -1, -1):
            for j in range(round(w/2)-20, round(w/2)+20):
                if img[i, j] == 255:
                    seed.append((j, i))
                    flag = 1
            if flag:
                initial_seed = seed[int(len(seed) / 2)]
                break
    else:
        print('Invalid side')

    return initial_seed, direction


def region_grow(img, side, th=1):
    # get initial seed
    initial_seed, direction = get_initial_seed(img, side)
    growing_pt = [0, 0]  # 试探生长点

    canvas = np.zeros(img.shape, np.uint8)  # 创建画板

    pt_stack = [initial_seed]
    canvas[initial_seed[1], initial_seed[0]] = 255
    seed_value = img[initial_seed[1], initial_seed[0]]

    while pt_stack:
        initial_seed = pt_stack.pop()

        # 分别对5/8个方向上的点进行生长
        for i in range(len(direction)):
            growing_pt[0] = initial_seed[0] + direction[i][0]
            growing_pt[1] = initial_seed[1] + direction[i][1]
            # 检查是否是边缘点
            if growing_pt[0] < 0 or growing_pt[1] < 0 or growing_pt[0] > \
                    img.shape[1]-1 or growing_pt[1] > img.shape[0] - 1:
                continue
            flag_grow = canvas[growing_pt[1], growing_pt[0]]     # 当前待生长点的灰度值

            if flag_grow == 0:
                current_value = img[growing_pt[1], growing_pt[0]]
                if abs(seed_value - current_value) < th:        # 与初始种子点像素差小于阈值
                    canvas[growing_pt[1], growing_pt[0]] = 255
                    pt_stack.append(growing_pt.copy())           # 新生长点入栈

    return canvas.copy()


def get_boundary(image):

    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    contours = measure.find_contours(img, 0.67, fully_connected='high')
    contours = [contours[i] for i in range(len(contours)) if len(contours[i]) > 100]
    contours_lst = []
    for i in range(len(contours)):
        for j in contours[i]:
            contours_lst.append(j)
    contours_lst = np.array(contours_lst)

    return contours_lst


def extract_vein_by_region_grow(edges_canny, image, threshold_perimeter, threshold_kernel_boundary):
    """
    edges_canny, image, threshold_perimeter, threshold_kernel_boundary -> vein, main_vein, vein_points, main_vein_points
    :param edges_canny: edges_canny.
    :param image: path_name of leave.
    :param threshold_perimeter: tolerated threshold of perimeter of contours, to get rid of the fractions of boundary.
    :param threshold_kernel_boundary: width of dilated boundary, to get rid of the boundary.
    :return: vein, main_vein, vein_points, main_vein_points.
    """
    # cut out boundary
    boundary = get_boundary(image)
    canvas_boundary = np.zeros(edges_canny.shape[:2], dtype=np.uint8)
    for i in boundary:
        canvas_boundary[int(i[0]), int(i[1])] = 255
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, threshold_kernel_boundary)
    canvas_boundary = cv2.dilate(canvas_boundary, kernel_boundary)  # 膨胀后的边框
    opened = cv2.bitwise_or(edges_canny, canvas_boundary)
    res_all = region_grow(opened, 'all')
    vein = cv2.subtract(res_all, canvas_boundary)

    h, w = vein.shape

    vein[:, round(w / 2) - 20:round(w / 2) + 20], contours = cv2.findContours(vein[:, round(w/2)-20:round(w/2)+20], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < 10]   # 删短周长的区域
    cv2.fillPoly(vein[:, round(w/2)-20:round(w/2)+20], small_perimeters, 0)

    vein_end = [0, 0]
    start_point_prev = [-1, -1]
    for end_idx in range(len(vein[::-1, round(w/2)-20:round(w/2)+20])):
        if vein[end_idx, :].any():
            for j in range(len(vein[end_idx, round(w/2)-20:round(w/2)+20])):
                if vein[:, round(w/2)-20:round(w/2)+20][end_idx][j] == 255:
                    vein_end = [end_idx, j+1]
    for i in range(len(vein[:vein_end[0], round(w/2)-20:round(w/2)+20])):
        if i != 0:
            if start_point and end_point and i in list(range(0, end_point[0])):
                continue
        start_point = []
        end_point = []
        flag_end = 'go'
        flag_continue_for_i = 'go'

        for j in range(len(vein[0:vein_end[0], round(w/2)-20:round(w/2)+20])):
            if flag_end == 'brk':
                break
            if vein[:, round(w/2)-20:round(w/2)+20][j].any() and start_point == []:
                if not vein[:, round(w/2)-20:round(w/2)+20][j+1].any():
                    for k in range(len(vein[:, round(w/2)-20:round(w/2)+20][j])):
                        if vein[:, round(w/2)-20:round(w/2)+20][j][k] == 255:
                            start_point = [j, (k+round(w/2)-20)+1]
                            if start_point[0] == start_point_prev[0]:
                                flag_continue_for_i = 'cnt'
                            start_point_prev = start_point.copy()
                            break
            if flag_continue_for_i == 'cnt':
                break
            if not vein[:, round(w/2)-20:round(w/2)+20][j].any() and start_point != [] and end_point == []:
                if vein[:, round(w/2)-20:round(w/2)+20][j+1].any():
                    for k in range(len(vein[:, round(w/2)-20:round(w/2)+20][j])):
                        if vein[:, round(w/2)-20:round(w/2)+20][j+1][k] == 255:
                            end_point = [j+1, (k+round(w/2)-20)+1]
                            flag_end = 'brk'
                            break
            else:
                continue
        # get points end
        if not start_point or not end_point or flag_continue_for_i == 'cnt':
            continue

        canny_threshold_enhanced_locally = [30, 60]

        vein_enhanced_locally = image[start_point[0]:end_point[0],
                                     min(start_point[1], end_point[1]):max(start_point[1], end_point[1])+1]

        edge_enhanced_locally = cv2.Canny(vein_enhanced_locally, *canny_threshold_enhanced_locally, apertureSize=3)

        white_pixel_percentage = list(edge_enhanced_locally.ravel() == 255).count(1) /\
                                 len(list(edge_enhanced_locally.ravel()))
        start_point_check = [-1, -1]
        counter_canny_adjustment = 0
        counter_prevent_dead_loop = 2
        white_pixel_percentage_prev = 0

        while not 1/40 < white_pixel_percentage < 1/20 and (start_point_check == [-1, -1] or
                                                                    start_point_check[1] >= start_point[0]):
            if not counter_prevent_dead_loop:

                break

            if white_pixel_percentage <= 1/40:
                canny_threshold_enhanced_locally = [canny_threshold_enhanced_locally[0] - 1,
                                                    canny_threshold_enhanced_locally[1] - 1]
            else:
                canny_threshold_enhanced_locally = [canny_threshold_enhanced_locally[0] + 1,
                                                    canny_threshold_enhanced_locally[1] + 1]

            edge_enhanced_locally = cv2.Canny(vein_enhanced_locally, *canny_threshold_enhanced_locally, apertureSize=3)

            white_pixel_percentage = (np.sum(edge_enhanced_locally==1) /
                                         len(list(edge_enhanced_locally.ravel())))
            if white_pixel_percentage == white_pixel_percentage_prev:
                counter_prevent_dead_loop -= 1
            white_pixel_percentage_prev = white_pixel_percentage

            counter_canny_adjustment += 1

            for k in range(len(vein[:, round(w / 2) - 20:round(w / 2) + 20])):
                if vein[:, round(w / 2) - 20:round(w / 2) + 20][k].any():
                    if not vein[:, round(w / 2) - 20:round(w / 2) + 20][min(vein.shape[0]-1, k + 1)].any():
                        for j in range(len(vein[:, round(w / 2) - 20:round(w / 2) + 20][k])):
                            if vein[:, round(w / 2) - 20:round(w / 2) + 20][k][j] == 255:
                                start_point_check = [k, j + 1]
                                break

        vein[start_point[0]:end_point[0], min(start_point[1], end_point[1]):max(start_point[1], end_point[1])+1] = \
            edge_enhanced_locally

    vein[:, round(w / 2) - 20:round(w / 2) + 20], contours= \
        cv2.findContours(vein[:, round(w/2)-20:round(w/2)+20], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < 10]   # 删短周长的区域
    cv2.fillPoly(vein[:, round(w/2)-20:round(w/2)+20], small_perimeters, 0)

    vein, contours = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < threshold_perimeter]   # 删短周长的区域
    cv2.fillPoly(vein, small_perimeters, 0)

    res_top = region_grow(vein, 'top')
    kernel_main_vein = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res_top = cv2.dilate(res_top, kernel_main_vein)
    res_top = cv2.dilate(res_top, kernel_main_vein)
    main_vein = cv2.bitwise_and(vein, res_top)

    # save points
    vein_points = []
    for i in range(vein.shape[0]):
        for j in range(vein.shape[1]):
            if vein[i, j] == 255:
                vein_points.append([i, j])
    main_vein_points = []
    for i in range(main_vein.shape[0]):
        for j in range(main_vein.shape[1]):
            if main_vein[i, j] == 255:
                main_vein_points.append([i, j])
    main_vein_points = np.array(main_vein_points)


    return vein, main_vein, vein_points, main_vein_points

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

    image_resolution = 800
    
    # Коэффициент масштаба
    scale_factor = 0.017

    # Применяем коэффициент масштаба к размерам объекта
    length_cm = length * scale_factor
    width_cm = width * scale_factor



    scaled_length = length / conversion_factor
    scaled_width = width / conversion_factor
    scaled_area = contour_area / (conversion_factor ** 2)


    M = cv2.moments(largest_contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])

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
    return scaled_length,scaled_width ,scaled_area , fluctuating_asymmetry

def canny_edges(image):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_equalized = clahe.apply(blurred)
    h, w = img_equalized.shape
    img_GB = cv2.bilateralFilter(image, 3, 50, 50)
    canny_threshold_common = [40, 100]
    canny_threshold_enhanced = [30, 60]
    edge_canny_up = cv2.Canny(img_GB[:round(h/6), round(w/2)-20:round(w/2)+20], *canny_threshold_common, apertureSize=3)
    edge_canny_middle = cv2.Canny(img_GB[round(h/6):round(h/2), round(w/2)-20:round(w/2)+20],
                                  *canny_threshold_enhanced, apertureSize=3)

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
    _, binary = cv2.threshold(edge_equalized, 0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)

    cv2.imshow('Binary Image', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge_equalized

def fractal_dimension(image):
    mask = mask_leaf(image)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    # print(f'Фрактальная размерность: {fractal_dimension}')
    return fractal_dimension


image = cv2.imread('data/leaf.png')
