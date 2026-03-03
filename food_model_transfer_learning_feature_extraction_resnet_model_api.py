from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model("food_model_transfer_learning_feature_extraction_resnet_model.keras")

# Class names (harus sama seperti training)
class_names = [
    "chicken_curry",
    "chicken_wings",
    "fried_rice",
    "grilled_salmon",
    "hamburger",
    "ice_cream",
    "pizza",
    "ramen",
    "steak",
    "sushi"
]

# ==============================
# Preprocess Image
# ==============================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# API Endpoint
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image)
    confidence = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]

    # OPTIONAL true label (jika dikirim dari Flutter)
    true_label = request.form.get("true_label", "Unknown")

    description = (
        f"Model memprediksi bahwa gambar ini adalah '{predicted_class}' "
        f"dengan tingkat kepercayaan {confidence*100:.2f}%. "
        f"Label sebenarnya adalah '{true_label}'."
    )

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "true_label": true_label,
        "description": description,
        "image_base64": img_base64
    })


if __name__ == "__main__":
    app.run(debug=True)