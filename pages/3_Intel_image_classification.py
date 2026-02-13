import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
from io import BytesIO
import time
import os

# --- Настройка страницы ---
st.set_page_config(page_title="Intel Image Classification", layout="wide")
st.title("🖼️ Intel Image Classification")
st.markdown("Загрузи изображения или вставь ссылку — модель определит тип сцены!")

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(
        torch.load("models/intel_model.pt", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

try:
    model = load_model()
    CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
except Exception as e:
    st.error(f"❌ Не удалось загрузить модель: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Вспомогательная функция предсказания ---
def predict_image(image):
    start_time = time.time()
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        predicted_class = CLASS_NAMES[idx.item()]
        conf_percent = confidence.item() * 100
    inference_time = time.time() - start_time
    return predicted_class, conf_percent, inference_time

# --- Вкладки: файлы vs URL ---
tab1, tab2 = st.tabs(["📁 Загрузить файлы", "🔗 По ссылке"])

# --- Вкладка 1: Множественная загрузка файлов ---
with tab1:
    uploaded_files = st.file_uploader(
        "Выбери одно или несколько изображений (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        cols = st.columns(min(3, len(uploaded_files)))
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                pred, conf, inf_time = predict_image(image)
                
                with cols[i % 3]:
                    st.image(image, use_container_width=True)
                    st.markdown(f"**Предсказание**: `{pred}`")
                    st.markdown(f"**Уверенность**: {conf:.1f}%")
                    st.caption(f"⏱️ {inf_time*1000:.1f} мс")
            except Exception as e:
                st.error(f"Ошибка при обработке {uploaded_file.name}: {e}")

# --- Вкладка 2: Загрузка по URL ---
with tab2:
    url = st.text_input("Вставь прямую ссылку на изображение (должна заканчиваться на .jpg / .png)")
    if url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            pred, conf, inf_time = predict_image(image)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Изображение из URL", use_container_width=True)
            with col2:
                st.success(f"**Предсказание**: `{pred}`")
                st.info(f"**Уверенность**: {conf:.1f}%")
                st.metric("Время инференса", f"{inf_time*1000:.1f} мс")
        except Exception as e:
            st.error(f"Не удалось загрузить изображение по ссылке: {e}")

# --- Подсказка ---
st.markdown("---")
st.caption("💡 Поддерживаемые классы: buildings, forest, glacier, mountain, sea, street")