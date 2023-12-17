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

    # Обработка изображения
    length, width, contour_area,assymetry = utils.measure_extract(image_array)
    class_leaf = utils.classfier_leaf(image_array)
    # Отображение обработанного изображения
    st.image(image, caption="Изображение")

    # Отображение характеристик изображения
    st.write("Длина: ", round(length,3))
    st.write("Ширина: ", round(width,3))
    st.write("Площадь: ", round(contour_area,3))
    st.write("флуктуирующая асимметрия листа: ", round(assymetry, 3))
    st.write("Класс листа : ", class_leaf)