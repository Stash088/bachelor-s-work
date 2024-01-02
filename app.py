import streamlit as st
from PIL import Image
import numpy as np
import cv2
import utils


# Streamlit web application
st.title("Приложение для фенотипирования ")
uploaded_file = st.file_uploader("Загрузка изображения", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Преобразование загруженного файла в объект изображения
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    # Обработка изображения
    length, width, contour_area,assymetry = utils.measure_extract(image_array)
    _, _, _, edge_equalized = utils.local_enhancement(image_rgb)
    vein, main_vein, vein_points, main_vein_points = utils.extract_vein_by_region_grow(edge_equalized, image_array, 150, (15, 15))
    class_leaf = utils.classfier_leaf(image_array)
    venation_leaf = utils.venation_leaf(image_array)
    # Отображение обработанного изображения
    st.image(image, caption="Изображение после детектора Canny")
    st.image(edge_equalized, caption="Изображение после детектора Canny")
    st.image(vein, caption="Изображение после детектора Canny")
 
    # Отображение характеристик изображения
    st.write("Длина: ", round(length,3))
    st.write("Ширина: ", round(width,3))
    st.write("Площадь: ", round(contour_area,3))
    st.write("флуктуирующая асимметрия листа: ", round(assymetry, 3))
    st.write("Класс листа : ", class_leaf)
    st.write("Венозность листа : ", venation_leaf)