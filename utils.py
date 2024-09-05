# utils.py

from feature_extraction import extract_features_from_video
import numpy as np

def load_videos_and_labels(real_videos, fake_videos):
    X = []
    y = []

    for video in real_videos:
        features = extract_features_from_video(video)
        X.extend(features)
        y.extend([0] * len(features))  # 0 for real

    for video in fake_videos:
        features = extract_features_from_video(video)
        X.extend(features)
        y.extend([1] * len(features))  # 1 for fake

    return np.array(X), np.array(y)

