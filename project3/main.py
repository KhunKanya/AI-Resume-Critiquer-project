import cv2               # for image processing
import numpy as np      # for numerical operations
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import( 
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)  # for deep learning model
from PIL import Image

# Load the pre-trained MobileNetV2 model
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Classify the image using the model
def classify_image(model, image):
    try:
        preprocess_image = preprocess_image(image, model)
        predictions = model.predict(preprocess_image)
        decode_predictions = decode_predictions(predictions, top=3)[0]

        return decode_predictions
    except Exception as e:
        st.error(f"Error classification image: {str(e)}")
        return None

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🚀", layout="centered")

    st.title("🚀 AI Image Classifier")
    st.write("Upload an image, and the AI model will classify it for you!")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       image = st.image(
           uploaded_file, caption="Uploaded Image.", use_column_width=True
       )
       btn = st.button("Classify Image")

       if btn:
           with st.spinner("Classifying..."):
               image = Image.open(uploaded_file)
               predictions = classify_image(image)

               if predictions:
                   st.success("redictions")
                   for _, label, score in predictions: (0, "car", 0.9)
                   st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()                 