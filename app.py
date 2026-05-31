import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import base64
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "crop_disease_model.h5"
)

HOME_BG = os.path.join(
    BASE_DIR,
    "backgrounds",
    "home_bg.jpg"
)

PREDICT_BG = os.path.join(
    BASE_DIR,
    "backgrounds",
    "predict_bg.jpg"
)

HISTORY_BG = os.path.join(
    BASE_DIR,
    "backgrounds",
    "history_bg.jpg"
)

HISTORY_FOLDER = os.path.join(
    BASE_DIR,
    "history"
)

os.makedirs(HISTORY_FOLDER, exist_ok=True)

HISTORY_CSV = os.path.join(
    HISTORY_FOLDER,
    "prediction_history.csv"
)

HISTORY_EXCEL = os.path.join(
    HISTORY_FOLDER,
    "prediction_history.xlsx"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================================================
# CLASS LABELS
# =====================================================

class_names = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Healthy Apple",
    "Healthy Blueberry",
    "Cherry Powdery Mildew",
    "Healthy Cherry",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Healthy Corn",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Leaf Blight",
    "Healthy Grape",
    "Orange Citrus Greening",
    "Peach Bacterial Spot",
    "Healthy Peach",
    "Pepper Bacterial Spot",
    "Healthy Pepper",
    "Potato Early Blight",
    "Potato Late Blight",
    "Healthy Potato",
    "Healthy Raspberry",
    "Healthy Soybean",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Healthy Strawberry",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Tomato"
]

# =====================================================
# BACKGROUND FUNCTION
# =====================================================

def set_background(image_file):

    try:

        with open(image_file, "rb") as file:
            encoded = base64.b64encode(
                file.read()
            ).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"Background image error: {e}"
        )

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction History"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if menu == "Home":

    set_background(HOME_BG)

    st.title("🌱 Crop Disease Detection System")

    st.markdown(
        """
        Upload a crop leaf image and predict the disease using Deep Learning.
        """
    )

    st.markdown(
        "<h4>Upload Image <span style='color:red'>*</span></h4>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "gif",
            "tiff",
            "webp"
        ],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    col1, col2 = st.columns(2)

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    with col1:

        if st.button("🔍 Predict Crop Disease"):

            if uploaded_file is None:

                st.warning(
                    "⚠️ Please upload an image!!"
                )

            else:

                set_background(PREDICT_BG)

                image = Image.open(
                    uploaded_file
                )

                st.image(
                    image,
                    caption="Uploaded Image",
                    width=350
                )

                image = image.convert("RGB")

                img = image.resize(
                    (224, 224)
                )

                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                img_array = img_array / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                confidence = (
                    np.max(prediction)
                    * 100
                )

                disease_name = (
                    class_names[
                        predicted_class
                    ]
                )

                st.success(
                    f"✅ Predicted Disease: {disease_name}"
                )

                st.info(
                    f"📊 Confidence Score: {confidence:.2f}%"
                )

                new_record = pd.DataFrame({

                    "Date & Time":
                    [datetime.now()],

                    "Disease":
                    [disease_name],

                    "Confidence (%)":
                    [round(confidence, 2)]

                })

                if os.path.exists(
                    HISTORY_CSV
                ):

                    old_df = pd.read_csv(
                        HISTORY_CSV
                    )

                    updated_df = pd.concat(
                        [
                            old_df,
                            new_record
                        ],
                        ignore_index=True
                    )

                    updated_df.to_csv(
                        HISTORY_CSV,
                        index=False
                    )

                else:

                    new_record.to_csv(
                        HISTORY_CSV,
                        index=False
                    )

    # =====================================================
    # CLEAR BUTTON
    # =====================================================

    with col2:

        if st.button("🗑 Clear"):

            if uploaded_file is None:

                st.warning(
                    "⚠️ Please upload an image!!"
                )

            else:

                st.session_state.uploader_key += 1

                st.rerun()

# =====================================================
# HISTORY PAGE
# =====================================================

elif menu == "Prediction History":

    set_background(HISTORY_BG)

    st.title(
        "📊 Prediction History"
    )

    if os.path.exists(
        HISTORY_CSV
    ):

        df = pd.read_csv(
            HISTORY_CSV
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        df.to_excel(
            HISTORY_EXCEL,
            index=False
        )

        with open(
            HISTORY_EXCEL,
            "rb"
        ) as file:

            st.download_button(
                label="⬇ Download Excel",
                data=file,
                file_name="prediction_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:

        st.warning(
            "No prediction history available."
        )