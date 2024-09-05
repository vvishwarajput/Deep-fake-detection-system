import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained VGG16 model
model = VGG16(include_top=False, input_shape=(224, 224, 3), pooling='avg')

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to 224x224 pixels
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    return img_array

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // num_frames)
    frames = []
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def extract_features_from_video(video_path):
    frames = extract_frames(video_path)
    features = [model.predict(preprocess_frame(frame)).flatten() for frame in frames]
    return np.array(features).reshape(-1, features[0].size)  # Ensure 2D shape

