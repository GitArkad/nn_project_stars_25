import streamlit as st
import torch
import time
import requests
from io import BytesIO
from PIL import Image
from torchvision.models import ResNet18_Weights

from models.model_blood_cells import load_model
st.set_page_config(page_title="Классификация клеток крови")

# ===================== КОНСТАНТЫ =====================

MODEL_PATH = "models/blood_cells.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

CLASS_NAMES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# ===================== ЗАГРУЗКА МОДЕЛИ =====================

@st.cache_resource
def get_model():
    model = load_model(MODEL_PATH)
    model.to(DEVICE)
    return model


# ===================== ПРЕДСКАЗАНИЕ =====================

def predict_image(model, image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    elapsed = time.perf_counter() - start

    return (
        CLASS_NAMES[pred_class.item()],
        confidence.item(),
        elapsed
    )


# ===================== ЗАГРУЗКА ИЗ URL =====================

def load_from_url(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


# ===================== СТРАНИЦА =====================

def render():
    st.title("Классификация клеток крови")

    model = get_model()


    # -------- Загрузка изображений --------
    st.subheader("Загрузка изображений")

    uploaded_files = st.file_uploader(
        "Загрузите одно или несколько изображений",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    url = st.text_input("Или вставьте ссылку на изображение")

    images = []

    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            images.append(image)

    if url:
        try:
            image = load_from_url(url)
            images.append(image)
        except Exception:
            st.error("Не удалось загрузить изображение по ссылке")

    # -------- Кнопка запуска --------
    if images and st.button("Классифицировать"):

        st.subheader("Результаты")

        total_time = 0

        for img in images:

            st.image(img, use_container_width=True)

            with st.spinner("Модель обрабатывает изображение..."):
                pred, conf, elapsed = predict_image(model, img)

            total_time += elapsed

            st.write(f"Предсказание: **{pred}**")
            st.write(f"Уверенность: **{conf:.4f}**")
            st.write(f"Время ответа модели: **{elapsed:.4f} секунд**")

            st.divider()

        st.success(f"Общее время обработки: {total_time:.4f} секунд")



if __name__ == "__main__":
    render()
