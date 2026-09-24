# 🌿 AI-Based Plant Disease Detection and Classification

An AI-based plant disease detection system developed using **Deep Learning, Transfer Learning, and EfficientNetB0**. The system classifies plant leaf images into **15 healthy and diseased categories** across **Tomato, Potato, and Pepper Bell** crops.

The trained model is integrated with a **Flask web application**, allowing users to upload a leaf image and receive the predicted disease, confidence score, and corresponding treatment recommendation.

## 🎯 Project Objectives

- Automatically detect plant diseases from leaf images.
- Classify images into 15 disease/healthy categories.
- Use EfficientNetB0 with transfer learning and fine-tuning.
- Provide confidence-based disease predictions.
- Recommend treatments for detected diseases.
- Provide an easy-to-use Flask web interface.

## 🧠 Model Architecture

The project uses **EfficientNetB0**, pre-trained on ImageNet, as the feature extraction backbone.

Architecture:

Input Image (224×224×3)  
↓  
EfficientNetB0  
↓  
Global Average Pooling  
↓  
Dropout (0.35)  
↓  
Dense Layer (256, ReLU)  
↓  
Dropout (0.25)  
↓  
Dense Layer (15, Softmax)  
↓  
Disease Prediction

## 📊 Dataset

The model was trained using the **PlantVillage dataset**.

- Total Images: **20,637**
- Number of Classes: **15**
- Crops: **Tomato, Potato, Pepper Bell**
- Training Split: **70%**
- Validation Split: **15%**
- Test Split: **15%**

## 🔬 Training Strategy

Training was performed in two stages:

### Stage 1 — Transfer Learning
The EfficientNetB0 backbone was frozen and the custom classification head was trained.

### Stage 2 — Fine-Tuning
The final 20 layers of EfficientNetB0 were unfrozen and trained using a lower learning rate.

## 📈 Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 97.33% |
| Precision | 97.40% |
| Recall | 97.33% |
| F1-Score | 97.33% |

## 🌱 Supported Classes

The system supports:

- Pepper Bell — Bacterial Spot
- Pepper Bell — Healthy
- Potato — Early Blight
- Potato — Late Blight
- Potato — Healthy
- Tomato — Bacterial Spot
- Tomato — Early Blight
- Tomato — Late Blight
- Tomato — Leaf Mold
- Tomato — Septoria Leaf Spot
- Tomato — Spider Mites
- Tomato — Target Spot
- Tomato — Yellow Leaf Curl Virus
- Tomato — Mosaic Virus
- Tomato — Healthy

## 💻 Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Transfer Learning
- Scikit-learn
- Flask
- HTML/CSS
- NumPy
- Matplotlib

## 📁 Project Structure

```text
Plant-Disease-detection/
│
├── app.py
├── train.py
├── disease_info.py
├── requirements.txt
├── README.md
│
├── models/
│   └── crop_disease_model.keras
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
└── dataset/
    ├── train/
    ├── valid/
    └── test/

🚀 How It Works
User uploads a plant leaf image.
Image is resized to 224×224 pixels.
EfficientNetB0 extracts visual features.
The trained classifier predicts one of 15 classes.
The application calculates the prediction confidence.
Disease information is retrieved.
The predicted disease and treatment recommendation are displayed.

⚙️ Installation

Clone the repository:

git clone <your-repository-url>
cd Plant-Disease-detection-

Install dependencies:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open the local address displayed by Flask in your browser.

🔮 Future Scope
Mobile application development
Real-time disease detection using camera feeds
Support for additional crops and diseases
Disease severity estimation
Multilingual support
IoT sensor integration
Drone-based crop monitoring
Weather-based disease risk prediction

👨‍💻 Author

Ashutosh Sahu
M.Tech — Computer Science Engineering
UPES

📌 Project Type

M.Tech Internship Project — AI-Based Plant Disease Detection and Classification using EfficientNetB0
