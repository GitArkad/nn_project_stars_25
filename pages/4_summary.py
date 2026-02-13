import streamlit as st
from PIL import Image
import os

# Путь к папке, где хранятся ваши сохраненные графики
IMG_DIR = "images"

def show_summary_page():
    st.title("📊 Сводная аналитика по всем моделям")
    st.info("Здесь собраны результаты обучения нейросетей для трех различных задач классификации.")

    # Создаем горизонтальные вкладки
    tab_sport, tab_blood, tab_nature = st.tabs([
        "⚽ Виды спорта", 
        "🔬 Клетки крови", 
        "🏞️ Природные сцены"
    ])

    # --- 1. РАЗДЕЛ: ВИДЫ СПОРТА (100 классов) ---
    with tab_sport:
        st.header("Классификация спорта")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Модель", "ResNet18")
        col2.metric("Разморозка слоев", "L3, L4, FC")       
        col3.metric("Время обучения", "10 мин 6 сек")
        col4.metric("Эпох", "15")
        col5.metric("Точность (Accuracy)", "96%")        
        col6.metric("Weighted F1-Score", "0.9575")

        st.subheader("Распределение классов")        
        raspr_sport_img = os.path.join(IMG_DIR, "raspred_classes_sic100.png")
        if os.path.exists(raspr_sport_img):
            st.image(raspr_sport_img, use_container_width=True, caption="Распределение для 100 классов")
        else:
            st.warning(f"Файл {raspr_sport_img} не найден в папке images")

        st.subheader("Графики метрик")
        # Проверяем наличие файла перед выводом
        metric_sport_img = os.path.join(IMG_DIR, "grafic_metrics_sic100.png")
        if os.path.exists(metric_sport_img):
            st.image(metric_sport_img, use_container_width=True, caption="Динамика Loss и Accuracy для 100 классов")
        else:
            st.warning(f"Файл {metric_sport_img} не найден в папке images")

        st.subheader("Heatmap")        
        heatmap_sport_img = os.path.join(IMG_DIR, "heatmap_sic100_final.png")
        if os.path.exists(heatmap_sport_img):
            st.image(heatmap_sport_img, use_container_width=True, caption="Heatmap для 100 классов")
        else:
            st.warning(f"Файл {heatmap_sport_img} не найден в папке images")

    # --- 2. РАЗДЕЛ: КЛЕТКИ КРОВИ (Blood Cells) ---
    with tab_blood:
        st.header("Классификация клеток крови")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Модель", "ResNet18")
        col2.metric("Разморозка слоев", "L3, L4, FC")       
        col3.metric("Время обучения", "10 мин 6 сек")
        col4.metric("Эпох", "8")
        col5.metric("Точность (Accuracy)", "82%")        
        col6.metric("F1-Score", "0.82")

        st.subheader("Распределение классов")        
        raspr_blood_img = os.path.join(IMG_DIR, "blood_raspr.png")
        if os.path.exists(raspr_blood_img):
            st.image(raspr_blood_img, use_container_width=True, caption="Распределение для 4 классов")
        else:
            st.warning(f"Файл {raspr_blood_img} не найден в папке images")        

        st.subheader("Графики метрик")
        # Проверяем наличие файла перед выводом
        metric_blood_img = os.path.join(IMG_DIR, "blood_cells_f1_3.png")
        if os.path.exists(metric_blood_img):
            st.image(metric_blood_img, use_container_width=True, caption="Динамика Loss и F1")
        else:
            st.warning(f"Файл {metric_blood_img} не найден в папке images")

        st.subheader("Heatmap")        
        heatmap_blood_img = os.path.join(IMG_DIR, "blood_cells_matrix_3.png")
        if os.path.exists(heatmap_blood_img):
            st.image(heatmap_blood_img, use_container_width=True, caption="Heatmap для 4 классов")
        else:
            st.warning(f"Файл {heatmap_blood_img} не найден в папке images")

    # --- 3. РАЗДЕЛ: ПРИРОДНЫЕ СЦЕНЫ (Intel Image) ---
    with tab_nature:
        st.header("Intel Image Classification")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Модель", "ResNet18")
        col2.metric("Разморозка слоев", "L4, FC")       
        col3.metric("Время обучения", "10 мин 22 сек")
        col4.metric("Эпох", "5")
        col5.metric("Точность (Accuracy)", "90.1%")        
        col6.metric("F1-Score", "0.89")


        st.subheader("Метрики")        
        raspr_intel_img = os.path.join(IMG_DIR, "metrics_intel.png")
        if os.path.exists(raspr_intel_img):
            st.image(raspr_intel_img, use_container_width=True, caption="Метрики")
        else:
            st.warning(f"Файл {raspr_intel_img} не найден в папке images")

        st.subheader("Графики метрик")
        # Проверяем наличие файла перед выводом
        metric_intel_img = os.path.join(IMG_DIR, "graphic_intel.png")
        if os.path.exists(metric_intel_img):
            st.image(metric_intel_img, use_container_width=True, caption="Динамика Loss и Accuracy")
        else:
            st.warning(f"Файл {metric_intel_img} не найден в папке images")

        st.subheader("Heatmap")        
        heatmap_intel_img = os.path.join(IMG_DIR, "Intel_heatmap.png")
        if os.path.exists(heatmap_intel_img):
            st.image(heatmap_intel_img, use_container_width=True, caption="Heatmap для 6 классов")
        else:
            st.warning(f"Файл {heatmap_intel_img} не найден в папке images")

# Если запускаем этот файл напрямую (для тестов)
if __name__ == "__main__":
    show_summary_page()
