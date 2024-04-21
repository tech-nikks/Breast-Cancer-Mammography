import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import pydicom
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('my_model')

def dicom_to_png(dicom_path):
    # Load the DICOM file
    dicom_image = pydicom.read_file(dicom_path).pixel_array
    
    # Convert DICOM to PNG
    png_image = Image.fromarray(dicom_image.astype(np.uint8))
    
    # Resize the image to match the expected input shape
    png_image = png_image.resize((48, 48))
    
    # Convert the image to a numpy array and add the channel dimension
    image = np.array(png_image)
    image = np.expand_dims(image, axis=-1)
    
    return image

def predict_breast_cancer(model, dicom_path):
    # Convert DICOM to array that can be used by the model
    image = dicom_to_png(dicom_path)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)
    
    # Convert prediction to a human-readable label
    if prediction[0][0] >= 0.5:
        result = "Breast cancer present"
    else:
        result = "No breast cancer"
    
    return result, image[0]  # Return the image without the batch dimension

st.title("Breast Cancer Detection")

# File upload
uploaded_file = st.file_uploader("Choose a DICOM file", type="dcm")

if uploaded_file is not None:
    # Save the uploaded file to disk
    with open("uploaded_file.dcm", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict on the uploaded DICOM file
    prediction, image = predict_breast_cancer(model, "uploaded_file.dcm")
    st.write(f"Prediction: {prediction}")
    
    # Display the image without squeezing
    st.image(image, caption='Uploaded Image', use_column_width=True)