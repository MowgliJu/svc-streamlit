import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage import io, color

from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.decomposition import TruncatedSVD, PCA
import streamlit as st

# Заголовок приложения
st.title("SVD в Изображениях :)")
st.write("Загрузите свою картинку")

# Кнопка для загрузки картинки
# uploader_file = st.sidebar.file_uploader("Нажмите сюда для загрузки файла", type=["png", "jpg", "jpeg"])
uploader_file = st.file_uploader("Нажмите сюда для загрузки файла", type=["png", "jpg", "jpeg"])
if uploader_file is not None:
    # Загрузка изображения
    image = io.imread(uploader_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.write(f"Размер изображения: {image.shape}")

    # Выбор канала цвета из RGB
    num = st.number_input(
        "Введите номер для цвета (0 - красный, 1 - зеленый, 2 - синий):",
        min_value=0,
        max_value=2,
        format="%d",
        help="Введите целое число от 0 до 2.",
    )
    
    # Извлечение одного канала цвета
    image_num = image[:, :, num]
    st.image(image_num, caption=f'Канал {num}', use_column_width=True)

    # Применение SVD
    U, sing_vals, Vt = np.linalg.svd(image_num, full_matrices=False)

    # Создание диагональной матрицы
    sigma = np.diag(sing_vals)

    st.write(f"Форма U: {U.shape}, Форма Sigma: {sigma.shape}, Форма Vt: {Vt.shape}")

    # Минимальное значение для ввода k
    min_val = min(image.shape[0], image.shape[1])
    k_num = st.number_input(
        "Введите число k (количество компонентов):",
        min_value=1,
        max_value=min_val,
        format="%d",
        help=f"Введите целое число от 1 до {min_val}.",
    )
    # if st.button("Сделать изображение черно-белым"):
    #     # Преобразование изображения в градации серого
    #     image_gray = rgb2gray(image)
    #     st.image(image_gray, caption='Черно-белое изображение', use_column_width=True)
    # # Точечное усечение U, Sigma и Vt
    
    trunc_U = U[:, :k_num]
    trunc_sigma = sigma[:k_num, :k_num]
    trunc_Vt = Vt[:k_num, :]

    # Восстановление изображения с использованием k компонент
    reconstructed_image = trunc_U @ trunc_sigma @ trunc_Vt

    # Посмотрите результат
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(image_num, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title(f'Восстановленное изображение с k = {k_num} компонент')
    
    # Отображение графиков
    st.pyplot(fig)

    st.write(f'Доля k = {k_num} составляет {100 * k_num / len(sing_vals)}%')
    
else:
    st.warning("Пожалуйста, загрузите изображение.")