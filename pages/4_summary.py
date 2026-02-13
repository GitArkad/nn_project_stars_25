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
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Точность (Accuracy)", "94.2%")
        col2.metric("Время обучения", "1ч 40м")
        col3.metric("Dataset", "Blood-Cells")

        st.subheader("Матрица ошибок (Heatmap)")
        blood_img = os.path.join(IMG_DIR, "blood_heatmap.png")
        if os.path.exists(blood_img):
            st.image(blood_img, use_container_width=True, caption="Хитмап распределения ошибок по типам клеток")
        else:
            st.warning("Файл 'blood_heatmap.png' не найден")

    # --- 3. РАЗДЕЛ: ПРИРОДНЫЕ СЦЕНЫ (Intel Image) ---
    with tab_nature:
        st.header("Intel Image Classification")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Точность (Accuracy)", "91.8%")
        col2.metric("Время обучения", "2ч 50м")
        col3.metric("Заморозка весов", "Включена")

        st.subheader("Анализ обучения")
        nature_img = os.path.join(IMG_DIR, "intel_plots.png")
        if os.path.exists(nature_img):
            st.image(nature_img, use_container_width=True, caption="Результаты обучения ResNet50")
        else:
            st.warning("Файл 'intel_plots.png' не найден")

# Если запускаем этот файл напрямую (для тестов)
if __name__ == "__main__":
    show_summary_page()
