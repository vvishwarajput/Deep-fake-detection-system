from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features_from_video
import joblib
import numpy as np

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def save_classifier(clf, filename):
    joblib.dump(clf, filename)

def load_classifier(filename):
    return joblib.load(filename)

def classify_video(video_path, clf):
    features = extract_features_from_video(video_path)
    if features.size == 0:
        return "Unable to extract features"
    prediction = clf.predict([features.mean(axis=0)])
    return "Fake" if prediction[0] == 1 else "Real"
