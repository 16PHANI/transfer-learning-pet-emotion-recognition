🐶🐱 Transfer Learning Based Pet Emotion Recognition

A web-based system that detects cat and dog emotions (Happy, Sad, Angry, Scared, Relaxed, Surprised) using Transfer Learning Models and provides automatic pet care tips based on the detected emotion.

🌟 Project Overview

This project uses deep learning transfer learning model ResNet50 to analyze pet images and classify their emotional state.
The predictions are integrated into a complete web application built using:

Frontend: HTML, CSS, JavaScript

Backend: Python (Flask / Django)

Model: Transfer Learning (ResNet50 / etc.)

Dataset: Cats & Dogs Emotional Dataset (linked via Google Drive)

Model File: model.h5 (linked via Google Drive)
The system also includes multiple pages such as Home, About, Daily Routine, Pet Care Tips, Prediction, and Contact.
📌 Features
🖥️ Frontend Pages
1️⃣ Home Page

Overview of the project

Information about pet emotion recognition

Navigation to all other pages

2️⃣ About Page

Explains the project purpose

Transfer Learning explanation

Technologies used

3️⃣ Daily Routine Page

Suggested daily routines for cats and dogs

Feeding, grooming, and activity guidelines

4️⃣ Pet Care Tips Page

Shows care tips dynamically

Tips change based on detected emotion

Example:

Happy → Reward with treats

Angry → Give space and calm environment

Scared → Comfort and ensure safety

5️⃣ Prediction Page

User uploads an image

Backend model predicts emotion

Displays:

Pet type (Cat/Dog)

Emotion label

Confidence score

A care tip for that emotion

6️⃣ Contact Page

Contact details

Message form

Social media links

🤖 Model Details

Transfer Learning Models explored:
✔ ResNet50
✔ VGG16 / VGG19
✔ MobileNet / EfficientNet

Final model (model.h5) deployed for prediction

Trained on 4–6 emotion classes for pets

Achieves high accuracy for real-world images
🔧 Tech Stack
Frontend:

HTML

CSS

JavaScript

Backend:

Python

Flask / Django

TensorFlow / Keras

Other Tools:

GitHub

Google Drive (for dataset & model storage)

📁 Project Structure
pet_emotion/
│── static/
│   ├── css/
│   ├── js/
│   └── images/
│
│── templates/
│   ├── home.html
│   ├── about.html
│   ├── predict.html
│   ├── daily_routine.html
│   ├── care_tips.html
│   └── contact.html
│
│── model/
│   └── model.h5 (stored in Drive)
│
│── app.py
│── README.md
│── requirements.txt
📥 Download Dataset & Model
📌 Dataset (Google Drive)

🔗 Add your dataset link here

📌 Model File (model.h5)

🔗 https://drive.google.com/file/d/1lpGqbORF_MGPFQzE-bVG4DR-i51WKzLn/view?usp=sharing

▶️ How to Run the Project Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Place model.h5 in /model/ folder

(or update the path in app.py)

3️⃣ Start Server
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

📸 Sample Output

Predicted Emotion

Confidence Score

Care Suggestions

Pet Image Preview

🚀 Future Enhancements

Add real-time webcam prediction

Add more pets (birds, rabbits)

Mobile app version

More accurate multi-emotion recognition

🤝 Contributors
Basthati Ramya ramani
BOYINAPALLI PHANI SHANKAR                                                   
Bathula Naga Manoj                                                              
BHATTU JAGADEESH PRASAD                                                    
