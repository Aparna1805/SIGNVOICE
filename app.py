from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import pickle
import io
import pyttsx3
from PIL import Image
import os
import logging
from speech_process import recognize_speech_from_mic

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Load sign language model
with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/speech', methods=['GET', 'POST'])
def speech():
    result = ""
    if request.method == 'POST':
        result = recognize_speech_from_mic()
        return render_template('speech.html', result=result)
    return render_template('speech.html', result=result)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        try:
            engine = pyttsx3.init()  # Create a new engine instance per request
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return jsonify({"status": "success", "message": f"Spoken: {text}"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "No text provided"}), 400
@app.route('/sign', methods=['POST'])
def sign():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        print("🖐️ Landmarks sample:", landmarks[:6])  # Just print first few numbers
        print("🖐️ Detected Landmarks:", landmarks)  # 👈 add this
        prediction = model.predict([landmarks])[0]
        print("📣 Prediction:", prediction)  # 👈 add this

        return jsonify({'prediction': prediction})
    else:
        print("🚫 No hand detected")
        return jsonify({'prediction': 'No hand detected'})

@app.route('/sign-page')
def sign_page():
    return render_template('sign.html')

@app.route('/speech-page')
def speech_page():
    return render_template('speech.html')

@app.route('/exit', methods=['GET'])
def exit_page():
    print("Exit route triggered!") 
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
