from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
from feature_extraction import extract_features_from_video
from classifier import classify_video, train_classifier, load_classifier,save_classifier
from utils import load_videos_and_labels

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load or train your classifier
if os.path.exists('model.pkl'):
    clf = load_classifier('model.pkl')
else:
    real_videos = ['/home/vishwa/Downloads/video_20210603_192318.mp4',
    '/home/vishwa/Downloads/rv1.mov',
    '/home/vishwa/Downloads/rv2.mov']
    fake_videos = ['/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2F922f0119-fe30-4d7d-8ec9-3162879d30eb_seed873115324.mp4',
    '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2Fd0e09e9b-3a98-4f0d-9959-647af7be5245_seed458598333.mp4',
    '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2Fbb9c163e-ef1d-4ce8-9c95-55cf31c8b4e5_seed1152630677.mp4'
]
    X, y = load_videos_and_labels(real_videos, fake_videos)
    clf, _, _ = train_classifier(X, y)
    save_classifier(clf, 'model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        result = classify_video(filename, clf)
        return jsonify({"result": result})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
