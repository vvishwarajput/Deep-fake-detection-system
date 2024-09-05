from feature_extraction import extract_features_from_video
from classifier import train_classifier, evaluate_classifier, classify_video
import numpy as np

def load_videos_and_labels(real_videos, fake_videos):
    X = []
    y = []

    for video in real_videos:
        features = extract_features_from_video(video)
        X.append(features)
        y.extend([0] * features.shape[0])  # 0 for real videos

    for video in fake_videos:
        features = extract_features_from_video(video)
        X.append(features)
        y.extend([1] * features.shape[0])  # 1 for fake videos

    # Combine all features into a single 2D array
    X = np.vstack(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

# Paths to videos
real_videos = [
    '/home/vishwa/Downloads/video_20210603_192318.mp4',
    '/home/vishwa/Downloads/rv1.mov',
    '/home/vishwa/Downloads/rv2.mov'
]
fake_videos = [
    '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2F922f0119-fe30-4d7d-8ec9-3162879d30eb_seed873115324.mp4',
    '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2Fd0e09e9b-3a98-4f0d-9959-647af7be5245_seed458598333.mp4',
    '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2Fbb9c163e-ef1d-4ce8-9c95-55cf31c8b4e5_seed1152630677.mp4'
]

# Load videos and extract features
X, y = load_videos_and_labels(real_videos, fake_videos)

# Split data into training and testing sets and train the classifier
clf, X_test, y_test = train_classifier(X, y)

# Evaluate the classifier
evaluate_classifier(clf, X_test, y_test)

# Classify a new video
video_to_test = '/home/vishwa/Downloads/pixverse%2Fmp4%2Fmedia%2Fd0e09e9b-3a98-4f0d-9959-647af7be5245_seed458598333.mp4'
result = classify_video(video_to_test, clf)
print(f"The video is classified as: {result}")
