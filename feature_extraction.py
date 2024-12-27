# feature_extraction.py
import pandas as pd
import numpy as np

def extract_features(segment):
    features = {}
    features["MAV"] = np.mean(np.abs(segment))
    features["RMS"] = np.sqrt(np.mean(segment ** 2))
    features["WL"] = np.sum(np.abs(np.diff(segment)))
    features["ZC"] = np.sum(np.diff(np.sign(segment)) != 0)
    return features

def feature_extraction(data):
    feature_list = []
    labels = data["Gesture_Class"]
    for i, row in data.iterrows():
        segment = row[:-1].values
        features = extract_features(segment)
        features["Gesture_Class"] = labels.iloc[i]
        feature_list.append(features)
    return pd.DataFrame(feature_list)

if __name__ == "__main__":
    input_file = "dataset/emg_data.csv"
    segmented_data = pd.read_csv(input_file)
    features = feature_extraction(segmented_data)
    features_file = "features_emg_data.csv"
    features.to_csv(features_file, index=False)
    print(f"Özellikler çıkarıldı ve kaydedildi: {features_file}")
