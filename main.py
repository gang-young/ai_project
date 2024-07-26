import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = tf.keras.models.load_model("keras_Model.h5", compile=False)

# Load the labels with UTF-8 encoding
with open("labels.txt", "r", encoding="utf-8") as file:
    class_names = file.readlines()

st.image('firefly_edited.jpg', use_column_width=True)

# Streamlit application
st.title("환자 낙상사고 예방 서비스")
st.write("이미지를 업로드하세요")

st.sidebar.write("주소: 부산광역시, 12345")
st.sidebar.write("전화번호: +82-10-1234-5678")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and process the image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any extra whitespace/newlines
    confidence_score = prediction[0][index]

    # Display the result in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    with col2:
        st.write(f"**Class:** {class_name}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}")
