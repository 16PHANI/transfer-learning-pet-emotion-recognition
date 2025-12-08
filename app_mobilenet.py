from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- AVAILABLE MODELS ----------------
MODEL_PATHS = {
    "resnet50": {
        "path": "pet_emotion_resnet50.h5",
        "img_size": (224, 224)
    }
   
}

# ---------------- SELECT MODEL ----------------
model_name = "resnet50"   # change to 'cnn' or 'mobilenet' if needed
model_path = MODEL_PATHS[model_name]["path"]
IMG_SIZE = MODEL_PATHS[model_name]["img_size"]

# Load model
model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded model: {model_name.upper()} from {model_path}")

# ---------------- CLASS LABELS ----------------
class_labels = [
    'alert', 'angry', 'happy', 'hungry', 'relaxed', 'sad', 'scared', 'surprised'
]

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

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/daily_routine')
def daily_routine():
    return render_template('daily_routine.html')

# ---------------- PREDICTION FUNCTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('predict.html', prediction="❌ No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('predict.html', prediction="❌ No image selected")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Preprocess image
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Test-Time Augmentation (TTA)
        def tta_predict(model, img_array):
            flips = [
                img_array,
                tf.image.flip_left_right(img_array),
                tf.image.flip_up_down(img_array)
            ]
            preds = [model.predict(flip, verbose=0) for flip in flips]
            return np.mean(preds, axis=0)

        # Run prediction
        prediction_probs = tta_predict(model, img_array)
        prediction_probs = tf.nn.softmax(prediction_probs).numpy()

        predicted_class = class_labels[np.argmax(prediction_probs)]

        # Always show a definite category
        result_text = f"{predicted_class.capitalize()}"

        # Debug info
        print("\n---- Prediction Debug Info ----")
        print("Predicted:", predicted_class)
        print("All class probabilities:")
        for i, cls in enumerate(class_labels):
            print(f"{cls}: {prediction_probs[0][i]*100:.2f}%")
        print("--------------------------------\n")

    except Exception as e:
        result_text = f"❌ Error processing image: {str(e)}"

    return render_template('predict.html', prediction=result_text, image_path=file_path)

# ---------------- CARE TIPS ----------------
@app.route('/care_tips/<emotion>')
def care_tips(emotion):
    emotion = emotion.capitalize()
    tips = {
        'Happy': {
            "description": "🎉 Your pet is happy! Keep up the love and playtime.",
            "food": ["Provide favorite treats", "Maintain balanced diet", "Ensure hydration"],
            "activity": ["Daily playtime", "Interactive toys", "Cuddles"]
        },
        'Sad': {
            "description": "😢 Your pet seems sad. Spend time, cuddle, or go for a walk.",
            "food": ["Comfort food", "Extra treats occasionally"],
            "activity": ["Gentle walks", "Affection & attention", "Quiet environment"]
        },
        'Angry': {
            "description": "😠 Your pet looks angry. Give space and calm environment.",
            "food": ["Avoid overfeeding", "Stick to routine meals"],
            "activity": ["Interactive toys", "Mental stimulation", "Calm surroundings"]
        },
        'Scared': {
            "description": "😨 Your pet is scared. Reassure and provide safety.",
            "food": ["Small meals", "Avoid loud noises during feeding"],
            "activity": ["Safe space", "Gentle touch", "Slow exposure to triggers"]
        },
        'Surprised': {
            "description": "😲 Your pet is surprised! Monitor and reassure.",
            "food": ["Normal meals", "Treats for comfort"],
            "activity": ["Observe behavior", "Play gently", "Calm environment"]
        },
        'Relaxed': {
            "description": "😌 Your pet is relaxed. Maintain comfort and peace.",
            "food": ["Regular meals", "Plenty of water"],
            "activity": ["Quiet rest", "Gentle play if desired"]
        },
        'Alert': {
            "description": "👀 Your pet is alert. Keep them stimulated and engaged.",
            "food": ["Balanced meals", "Healthy snacks"],
            "activity": ["Training exercises", "Interactive toys", "Short walks"]
        },
        'Hungry': {
            "description": "🍖 Your pet is hungry. Feed appropriately and maintain schedule.",
            "food": ["Feed balanced meal", "Healthy treats", "Fresh water"],
            "activity": ["Short walks after meal", "Gentle play"]
        },
        'Normal': {
            "description": "😐 Your pet is in a neutral mood. Keep routine consistent.",
            "food": ["Regular diet", "Monitor water intake"],
            "activity": ["Routine walks", "Regular playtime"]
        }
    }

    selected_tip = tips.get(emotion, tips['Normal'])
    return render_template(
        'care_tips.html',
        emotion=emotion,
        tip=selected_tip["description"],
        food_tips=selected_tip["food"],
        activity_tips=selected_tip["activity"]
    )

# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
