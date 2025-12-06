import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import warnings 

warnings.filterwarnings('ignore')

# Load model sekali di awal
model = load_model('CNN House Plant.h5')

def predict_image(uploaded_file):
    # Simpan sementara di temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    img = image.load_img(temp_path, target_size=(150, 150))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img)[0][0]

    return pred >= 0.5  # True=Wilted, False=Healthy


st.title("Classification House Plant")
st.write("Upload an image of the plant leaf.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict_image(uploaded_file)
        if result:
            st.success("🌱 The Plant is **Wilted**")
        else:
            st.success("🌿 The Plant is **Healthy**")
