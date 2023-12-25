import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL = tf.keras.models.load_model("F:\\class\\DeepLearning\\potato leaf classification\\1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = (255, 255)

st.set_page_config(page_title="Potato Disease Classification", page_icon="🥔")

st.title("Potato Disease Classification")

st.write(
    "This application uses deep learning to classify potato leaf diseases. "
    "Upload an image of a potato leaf, and the model will predict whether it is affected by Early Blight, Late Blight, or if it's Healthy."
)

# Developer Information
st.sidebar.title("Developer: Hakim")
st.sidebar.write(
    "Hakim is a student studying BSPI on CST. Currently, he is exploring the field of machine learning. "
    "This project focuses on using deep learning for agricultural purposes."
)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        st.write("Classifying...")

        # Resize the image to the expected input shape
        image = image.resize(IMAGE_SIZE)
        img_array = np.array(image)
        img_batch = np.expand_dims(img_array, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")
