from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')
print("----------- model loaded ----------------")
# Preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to match model input shape
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize pixel values between 0 and 1
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# API route for predicting emotions
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    image = Image.open(image)

    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    emotions = ['happy', 'sad', 'angry', 'neutral']
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = emotions[predicted_class]
    confidence = float(predictions[0][predicted_class])

    return jsonify({'emotion': predicted_emotion, 'confidence': confidence})

@app.route("/",methods=["GET"])
def test():
	return  jsonify({"message" : "helloworld"})

if __name__ == '__main__':
    app.run()
