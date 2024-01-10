# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load and preprocess your dataset
# Assume your dataset is organized in a directory structure, adjust accordingly
data_directory = r"C:\Users\NIKHIL KUMAR REDDY\Downloads\archive.zip\PlantVillage"

# Use TensorFlow's ImageDataGenerator for data augmentation and preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Adjust the validation split based on your dataset
)

# Set up training and validation data generators
train_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specify 'training' for the training split
)

validation_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specify 'validation' for the validation split
)

# Load the pre-trained EfficientNetB3 model
base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# Add custom dense layers for classification
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
predictions = tf.keras.layers.Dense(7, activation='softmax')(x)  # 7 classes

# Create the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model on the validation set
validation_data = validation_generator.next()
validation_images, validation_labels = validation_data

validation_pred = model.predict(validation_images)
validation_pred_classes = np.argmax(validation_pred, axis=1)
validation_true_classes = np.argmax(validation_labels, axis=1)

# Calculate metrics
accuracy = accuracy_score(validation_true_classes, validation_pred_classes)
precision = precision_score(validation_true_classes, validation_pred_classes, average='weighted')
recall = recall_score(validation_true_classes, validation_pred_classes, average='weighted')
f1 = f1_score(validation_true_classes, validation_pred_classes, average='weighted')

# Streamlit UI
st.title("LeafClassifier: Plant Species Identification")

uploaded_file = st.file_uploader("Choose a plant image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    class_labels = {0: 'Crops', 1: 'Fruits', 2: 'Industrial Plants', 3: 'Medicinal Plants', 4: 'Nuts', 5: 'Tubers', 6: 'Vegetable Plants'}
    predicted_class = class_labels[predicted_class_index]

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
