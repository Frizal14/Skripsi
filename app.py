import os
import cv2
import gdown
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense

# -----------------------------------------------------------------------------
# FIX KOMPATIBILITAS KERAS V3 / TENSORFLOW 2.16+
# -----------------------------------------------------------------------------
class CompatibleDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Helper function untuk render gambar yang kompatibel di semua versi Streamlit
def safe_image(image_data, **kwargs):
    try:
        st.image(image_data, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image_data, use_column_width=True, **kwargs)

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Ikan & XAI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASSES = ['Archer Fish', 'Betta Fish', 'Blue tang', 'Clown Sword Trigger Fish', 'Yellow Tang']

# -----------------------------------------------------------------------------
# 2. DICTIONARY SEMUA MODEL (Non-TL & TL)
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "MobileNetV2 - Transfer Learning (Winner - 99.20%)": {
        "file_id": "1CZouFlsMkU4anU6pjz6w9v7KCGDgB0-w",
        "filename": "mobilenet_tl_best.h5",
        "acc": "99.20%",
        "type": "Transfer Learning (ImageNet)"
    },
    "VGG16 - Transfer Learning (94.71%)": {
        "file_id": "1nO0wNMwrZhEvwNVnlMgu5Qd4y2born9m",
        "filename": "vgg16_tl_best.h5",
        "acc": "94.71%",
        "type": "Transfer Learning (ImageNet)"
    },
    "VGG16 - Non-Transfer Learning (88.98%)": {
        "file_id": "1Efqj7QMK_HqC4uyd48wbfkyyEwlPhG4b",
        "filename": "vgg16_non_tl_best.h5",
        "acc": "88.98%",
        "type": "From Scratch (Non-TL)"
    },
    "MobileNetV2 - Non-Transfer Learning (Collapse ~20%)": {
        "file_id": "1TYMtCcufVBsV3x2rR7uLiH8oB3eNqN_Q",
        "filename": "mobilenet_non_tl_best.h5",
        "acc": "~20.00%",
        "type": "From Scratch (Non-TL)"
    }
}

# -----------------------------------------------------------------------------
# 3. SIDEBAR: PEMILIHAN MODEL DYNAMIC
# -----------------------------------------------------------------------------
st.sidebar.title("🐟 Pengaturan Model")
selected_model_key = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    list(MODEL_CONFIGS.keys())
)

selected_config = MODEL_CONFIGS[selected_model_key]
model_file_id = selected_config["file_id"]
model_filename = selected_config["filename"]

# Cache resource agar model hanya pernah di-load sekali ke RAM
@st.cache_resource(show_spinner=False)
def load_selected_model(file_id, filename):
    if not os.path.exists(filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=True)
    
    model = tf.keras.models.load_model(
        filename,
        custom_objects={'Dense': CompatibleDense},
        compile=False
    )
    return model

try:
    with st.spinner("🐟 Memuat model ke memori..."):
        model = load_selected_model(model_file_id, model_filename)
    st.sidebar.success(f"🐟 Model Aktif: {model_filename}")
except Exception as e:
    st.sidebar.error(f"🐟 Gagal memuat model: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.title("🐟 Detail Model Aktif")
st.sidebar.markdown(f"""
- **Tipe Pelatihan**: {selected_config['type']}
- **Akurasi Validasi**: {selected_config['acc']}
- **Ukuran Input**: 160 x 160 px
- **Metode XAI**: Occlusion Sensitivity
""")

# -----------------------------------------------------------------------------
# 4. FUNGSI EXPLAINABLE AI (OCCLUSION SENSITIVITY)
# -----------------------------------------------------------------------------
def generate_occlusion_heatmap(model_obj, img_array, patch_size=30, stride=15):
    h, w, _ = img_array.shape
    input_tensor = np.expand_dims(img_array, axis=0)
    
    preds = model_obj.predict(input_tensor, verbose=0)
    top_class = np.argmax(preds[0])
    baseline_prob = preds[0][top_class]
    
    heatmap = np.zeros((h, w))
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded_img = img_array.copy()
            occluded_img[y:y+patch_size, x:x+patch_size, :] = 0.5
            
            preds_occ = model_obj.predict(np.expand_dims(occluded_img, axis=0), verbose=0)
            drop = baseline_prob - preds_occ[0][top_class]
            heatmap[y:y+patch_size, x:x+patch_size] += drop

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_cm = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_cm = cv2.cvtColor(heatmap_cm, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = np.clip(0.6 * img_array + 0.4 * heatmap_cm, 0, 1)
    
    return heatmap_resized, overlay

# -----------------------------------------------------------------------------
# 5. ANTARMUKA UTAMA
# -----------------------------------------------------------------------------
st.title("🐟 Sistem Klasifikasi Spesies Ikan & Explainable AI")
st.write("Unggah citra ikan untuk mengidentifikasi spesies serta melihat area visual yang menjadi fokus keputusan model AI.")

st.markdown("---")

uploaded_file = st.file_uploader("Pilih file gambar ikan (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((160, 160))
    img_array = np.array(image_resized) / 255.0
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("🐟 Gambar Input")
        safe_image(image)
        
    with st.spinner("🐟 Menganalisis gambar..."):
        preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASSES[pred_idx]
        confidence = preds[pred_idx] * 100

    with col2:
        st.subheader("🐟 Hasil Prediksi")
        
        # Kartu indikator hasil prediksi
        m1, m2 = st.columns(2)
        m1.metric("Spesies Terdeteksi", pred_class)
        m2.metric("Tingkat Kepercayaan", f"{confidence:.2f}%")
        
        st.write("---")
        st.write("**Distribusi Probabilitas Kelas:**")
        for cls, prob in zip(CLASSES, preds):
            st.progress(float(prob), text=f"{cls}: {prob*100:.2f}%")

    st.markdown("---")
    
    # Section Explainable AI
    st.subheader("🐟 Visualisasi Explainable AI (Occlusion Sensitivity)")
    st.caption("Peta panas (heatmap) menunjukkan wilayah tubuh ikan yang paling berpengaruh terhadap hasil klasifikasi.")
    
    if st.button("Jalankan Analisis XAI", type="primary"):
        with st.spinner("🐟 Menghitung heatmap sensitivitas oklusi..."):
            heatmap, overlay = generate_occlusion_heatmap(model, img_array)
            
            xcol1, xcol2, xcol3 = st.columns(3)
            
            with xcol1:
                st.markdown("**Gambar Asli**")
                safe_image(image_resized)

            with xcol2:
                st.markdown("**Peta Heatmap**")
                safe_image(heatmap, clamp=True)

            with xcol3:
                st.markdown("**Overlay Heatmap**")
                safe_image(overlay)