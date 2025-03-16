import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained CNN model
model = load_model("traffic_sign_model.h5")

# Define class names (Modify this based on your dataset labels)
class_names = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60",
    "Speed Limit 70", "Speed Limit 80", "End of Speed Limit 80",
    "Speed Limit 100", "Speed Limit 120", "No Passing", "No Passing for trucks",
    "Right of way at intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles > 3.5 tons"
]

# Streamlit UI
st.title("🚦 Traffic Sign Recognition App")
st.write("Upload an image of a traffic sign, and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((32, 32))  # Resize to match CNN input
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # Display result
    st.subheader(f"🚀 Predicted Traffic Sign: {class_names[predicted_class]}")
    st.write(f"**Confidence Score:** {np.max(prediction) * 100:.2f}%")
