# 🌱 Crop Disease Detection Using Deep Learning

## 📌 Project Overview

Crop Disease Detection is a Deep Learning-based image classification project that identifies diseases in crop leaves from uploaded images. The system helps farmers and agricultural experts detect plant diseases at an early stage, enabling timely treatment and reducing crop losses.

This project uses Transfer Learning with MobileNetV2 and is trained on the New Plant Diseases Dataset (Augmented) from Kaggle.

---

## 🎯 Objectives

* Detect crop diseases from leaf images.
* Classify images into one of 38 disease categories.
* Provide disease prediction with confidence score.
* Maintain prediction history.
* Offer a user-friendly Streamlit web application.

---

## 📂 Dataset

Dataset: New Plant Diseases Dataset (Augmented)

Source:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### Dataset Statistics

* Training Images: 70,295
* Validation Images: 17,572
* Total Classes: 38

---

## 🏗️ Model Architecture

### Transfer Learning Model

* Base Model: MobileNetV2
* Input Shape: 224 × 224 × 3
* Global Average Pooling Layer
* Dense Layer (256 Neurons)
* Dropout Layer (0.5)
* Output Layer (38 Classes, Softmax)

### Optimizer

* Adam Optimizer

### Loss Function

* Categorical Crossentropy

### Evaluation Metric

* Accuracy

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* Streamlit
* NumPy
* Pandas
* Pillow (PIL)
* OpenPyXL

---

## 📁 Project Structure

```text
Crop_Disease_Detection_App/
│
├── app.py
├── README.md
├── requirements.txt
│
├── model/
│   └── crop_disease_model.h5
│
├── backgrounds/
│   ├── home_bg.jpg
│   ├── predict_bg.jpg
│   └── history_bg.jpg
│
├── history/
│   ├── prediction_history.csv
│   └── prediction_history.xlsx
│
└── screenshots/
```

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

* Image resizing to 224 × 224
* Pixel normalization (0–255 → 0–1)
* Data augmentation:

  * Rotation
  * Zoom
  * Width Shift
  * Height Shift
  * Horizontal Flip

These techniques improve model generalization and reduce overfitting.

---

## 🚀 Model Training

### Training Configuration

* Epochs: 5
* Batch Size: 32
* Optimizer: Adam
* Transfer Learning: MobileNetV2
* Early Stopping
* Reduce Learning Rate on Plateau

---

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Sample Accuracy

```text
Validation Accuracy ≈ 93%
```

---

## 💻 Streamlit Application Features

### Home Page

* Upload leaf image
* Predict disease
* Display confidence score
* Clear uploaded image

### Prediction Module

* Disease prediction
* Confidence percentage
* Image preview

### Prediction History

* Stores all predictions
* Download prediction history as Excel file

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/crop-disease-detection.git
```

### Step 2: Navigate to Project Folder

```bash
cd crop-disease-detection
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit Application

```bash
streamlit run app.py
```

---

## 🌿 Supported Crop Diseases

The model can identify diseases from:

* Apple
* Blueberry
* Cherry
* Corn (Maize)
* Grape
* Orange
* Peach
* Bell Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

Total Classes: 38

---

## 📸 Screenshots

### Home Page

Add screenshot here.

### Prediction Page

Add screenshot here.

### History Page

Add screenshot here.

---

## 🔮 Future Enhancements

* Real-time disease detection using mobile camera
* Disease treatment recommendations
* Fertilizer suggestions
* Cloud deployment
* Multi-language support

---

## 👩‍💻 Author

Swathi

Data Science & Generative AI Project

---

## 📜 License

This project is developed for educational and academic purposes.
