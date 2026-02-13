import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import time
import json
import os

# --- 1. ЗАГРУЗКА ДАННЫХ ---

@st.cache_data
def load_class_names(json_path):
    """Загружает названия классов из JSON файла."""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Приводим ключи к целым числам для сравнения с выходом модели
            return {int(k): v for k, v in data.items()}
    else:
        # Если файла нет, создаем временный список, чтобы код работал
        st.warning(f"Файл {json_path} не найден! Проверьте наличие classes.json в папке models.")
        return {i: f"Класс №{i}" for i in range(100)}

@st.cache_resource
def load_trained_model(model_path):
    """Загружает архитектуру и веса модели."""
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 100) 
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Настройки путей и объектов
MODEL_PATH = 'models/model_sic100.pt'
JSON_PATH = 'models/classes_sic100.json'

CLASS_LABELS = load_class_names(JSON_PATH)
model = load_trained_model(MODEL_PATH)

# Преобразования для ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. ИНТЕРФЕЙС ---

st.title("⚽ Классификатор изображений видов спорта")

# Хранилище в session_state (инициализируем, если пусто)
if 'images_archive' not in st.session_state:
    st.session_state.images_archive = []

tab1, tab2 = st.tabs(["📥 Загрузка", "🔍 Анализ"])

with tab1:
    files = st.file_uploader("Выберите фото", accept_multiple_files=True, key="uploader")
    url = st.text_input("Или вставьте ссылку")
    
    if st.button("Добавить в список"):
        if files:
            for f in files:
                st.session_state.images_archive.append(Image.open(f).convert('RGB'))
        if url:
            try:
                res = requests.get(url, timeout=5)
                st.session_state.images_archive.append(Image.open(BytesIO(res.content)).convert('RGB'))
            except:
                st.error("Не удалось загрузить по ссылке.")
        st.success(f"Фотографий в очереди: {len(st.session_state.images_archive)}")

with tab2:
    if not st.session_state.images_archive:
        st.info("Загрузите изображения во вкладке выше.")
    else:
        # Кнопки управления
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            start_analysis = st.button("🚀 НАЧАТЬ АНАЛИЗ", type="primary", use_container_width=True)
        with c_btn2:
            if st.button("🗑️ Очистить всё", use_container_width=True):
                st.session_state.images_archive = []
                st.rerun()

        # Цикл предсказания
        for i, img in enumerate(st.session_state.images_archive):
            st.write("---")
            col_img, col_res = st.columns([1, 1.5])
            
            with col_img:
                st.image(img, use_container_width=True, caption=f"Фото №{i+1}")
            
            with col_res:
                if start_analysis:
                    # Логика модели
                    start_t = time.time()
                    
                    input_tensor = preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        conf, idx = torch.max(probs, dim=1)
                    
                    end_t = time.time()
                    duration = (end_t - start_t) * 1000 # мс
                    
                    # ПРИВЯЗКА КЛАССА К НАЗВАНИЮ
                    class_id = int(idx.item())
                    # Берем название из словаря, если его нет — выводим ID
                    name = CLASS_LABELS.get(class_id, f"ID {class_id}")
                    
                    # ВЫВОД РЕЗУЛЬТАТОВ
                    st.success(f"### Результат: {name}")
                    st.metric("Точность", f"{conf.item():.2%}")
                    st.write(f"⏱ Время: {duration:.2f} мс")
                else:
                    st.write("Нажмите кнопку выше для запуска.")
