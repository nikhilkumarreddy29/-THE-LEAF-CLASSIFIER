import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
import streamlit as st
model = tf.keras.applications.EfficientNetB3(weights='imagenet')
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
def predict_plant_species(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0][0][1]
st.title("LeafClassifier: Plant Species Identification")
uploaded_file = st.file_uploader("Choose a plant image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict plant species
    plant_species = predict_plant_species(uploaded_file)
    
    st.success(f"Prediction: {plant_species}")
