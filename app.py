import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

from disease_info import disease_info


app = Flask(__name__)

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model 
model = tf.keras.models.load_model(
    "models/crop_disease_model.keras"
)

# Class Names
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="No image uploaded."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    # Load Image
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Match EfficientNet preprocessing used during training
    img_array = preprocess_input(img_array)

    # Prediction (deterministic inference)
    prediction = model(img_array, training=False).numpy()

    class_index = int(np.argmax(prediction))


    class_name = CLASS_NAMES[class_index]

    result = disease_info.get(
        class_name,
        {"disease": class_name, "cure": "No treatment info available."},
    )

    # Higher precision confidence for more stable display
    confidence = float(np.max(prediction) * 100.0)

    return render_template(
        "index.html",
        disease=result["disease"],
        cure=result["cure"],
        confidence=round(confidence, 4),
        image_path=f"uploads/{filename}",
    )


if __name__ == "__main__":
    app.run(debug=True)