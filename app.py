from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# MUST MATCH TRAINING ORDER
class_labels = ['alert', 'angry', 'happy', 'relaxed', 'sad']

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/daily_routine')
def daily_routine():
    return render_template('daily_routine.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('predict.html', prediction="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('predict.html', prediction="No image selected")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # Preprocess
    img = image.load_img(path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    emotion = class_labels[idx]
    confidence = preds[idx] * 100

    result = f"{emotion.capitalize()} ({confidence:.1f}%)"

    return render_template(
        'predict.html',
        prediction=result,
        image_path=path,
        emotion=emotion
    )

@app.route('/care_tips/<emotion>')
def care_tips(emotion):
    tips = {
        'Happy': {
            "description": "🎉 Your pet is happy!",
            "food": ["Balanced meals", "Fresh water"],
            "activity": ["Playtime", "Cuddles"]
        },
        'Sad': {
            "description": "😢 Your pet seems sad.",
            "food": ["Comfort food"],
            "activity": ["Gentle walks"]
        },
        'Angry': {
            "description": "😠 Your pet looks angry.",
            "food": ["Routine meals"],
            "activity": ["Calm environment"]
        },
        'Relaxed': {
            "description": "😌 Your pet is relaxed.",
            "food": ["Regular meals"],
            "activity": ["Rest"]
        },
        'Alert': {
            "description": "👀 Your pet is alert.",
            "food": ["Healthy snacks"],
            "activity": ["Training"]
        }
    }

    emotion = emotion.capitalize()
    tip = tips.get(emotion, tips['Happy'])

    return render_template(
        'care_tips.html',
        emotion=emotion,
        tip=tip["description"],
        food_tips=tip["food"],
        activity_tips=tip["activity"]
    )

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
