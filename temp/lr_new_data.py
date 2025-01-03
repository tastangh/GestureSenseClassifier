import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, lfilter
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Low-Pass Filter Function
def low_pass_filter(data, cutoff=50, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)


# 2. Advanced Feature Extraction
def advanced_feature_extraction(window):
    feature_vector = {
        "mean": window.mean(axis=0),
        "std": window.std(axis=0),
        "min": window.min(axis=0),
        "max": window.max(axis=0),
        "energy": np.sum(np.square(window), axis=0),
        "zero_crossing_rate": np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0),
        "dominant_frequency": np.argmax(np.abs(np.fft.fft(window, axis=0)), axis=0),
        "rms": np.sqrt(np.mean(np.square(window), axis=0)),
        "waveform_length": np.sum(np.abs(np.diff(window, axis=0)), axis=0),
    }
    return np.concatenate(list(feature_vector.values()))


# 3. Feature Extraction
def extract_features(data, channels, window_size=200):
    features = []
    labels = []
    for gesture in data["class"].unique():
        subset = data[data["class"] == gesture]
        for i in range(0, len(subset) - window_size, window_size):
            window = subset.iloc[i : i + window_size][channels].values
            features.append(advanced_feature_extraction(window))
            labels.append(gesture)
    return np.array(features), np.array(labels)


# 4. Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


# 5. Main Workflow
def main(file_path):
    # Load Data
    print("Loading data...")
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns if present
    if "label" in data.columns:
        data = data.drop(columns=["label"])
    
    # Filter EMG signals
    channels = [f"channel{i}" for i in range(1, 9)]
    for col in channels:
        data[col] = low_pass_filter(data[col].values)
    
    # Feature Extraction
    print("Extracting features...")
    features, labels = extract_features(data, channels)
    
    # Handle Imbalance with SMOTE
    print("Balancing dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    features, labels = smote.fit_resample(features, labels)
    
    # Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Scale Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    # Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, np.unique(labels), title="Logistic Regression Confusion Matrix")


if __name__ == "__main__":
    # Specify dataset path
    dataset_path = "dataset/EMG-data.csv"  # Update the path as needed
    main(dataset_path)
