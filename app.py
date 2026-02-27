import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
st.title("Brain Tumor Detection")

@st.cache_resource
def load_my_model():
    return load_model("tumor_detector.h5", compile=False)

try:
    model = load_my_model()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    confidence = pred * 100

    if pred > 0.5:
        label = "Tumor Detected"
    else:
        label = "No Tumor"

    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")