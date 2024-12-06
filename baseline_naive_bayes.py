"""
baseline_naive_bayes.py
----------------
This is some code used to 
"""

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from utils.preprocessing import extract_color_histogram

def load_images_and_labels(root_folder):
    data = []
    labels = []
    class_names = []
    for subdir in Path(root_folder).iterdir():
        if subdir.is_dir():
            class_name = subdir.name
            class_names.append(class_name)
            for img_file in subdir.glob("*.jpeg"):
                image = cv2.imread(str(img_file))
                if image is None:
                    print("oof")
                features = extract_color_histogram(image)
                data.append(features)
                labels.append(class_name)
    print(len(data))
    return np.array(data), np.array(labels), class_names

def main(root_folder):
    data, labels, class_names = load_images_and_labels(root_folder)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    # eval model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main('img')
