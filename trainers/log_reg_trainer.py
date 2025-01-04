import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


class LogRegTrainer:
    def __init__(self, file_path, channels, window_size=200, cutoff=(20, 450), sampling_rate=1000):
        self.file_path = file_path
        self.channels = channels
        self.window_size = window_size
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate

    @staticmethod
    def drop_columns(data, columns):
        return data.drop(columns=columns, errors="ignore")

    @staticmethod
    def drop_unmarked_class(data, class_column, unmarked_value=0):
        return data[data[class_column] != unmarked_value].reset_index(drop=True)

    @staticmethod
    def drop_na(data):
        return data.dropna().reset_index(drop=True)

    @staticmethod
    def apply_filter(signal, filter_type, cutoff, order=4, sampling_rate=1000):
        nyquist = 0.5 * sampling_rate
        if filter_type == "band":
            normalized_cutoff = [freq / nyquist for freq in cutoff]
            b, a = butter(order, normalized_cutoff, btype="band")
        else:
            raise ValueError("Only bandpass filter is supported.")
        return filtfilt(b, a, signal)

    def filter_all_channels(self, data):
        for channel in self.channels:
            data[channel] = self.apply_filter(data[channel].values, "band", self.cutoff, order=4, sampling_rate=self.sampling_rate)
        return data

    @staticmethod
    def advanced_feature_extraction(window):
        # Zaman alanı özellikleri
        mean_abs_value = np.mean(np.abs(window), axis=0)
        root_mean_square = np.sqrt(np.mean(np.square(window), axis=0))
        waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        variance = np.var(window, axis=0)
        integrated_emg = np.sum(np.abs(window), axis=0)

        # Frekans alanı özellikleri
        freq_spectrum = np.fft.fft(window, axis=0)
        power_spectral_density = np.abs(freq_spectrum) ** 2

        # Mean frequency düzeltildi: kanal bazında işlem yapılmalı
        mean_frequency = np.array([
            np.sum(np.arange(len(psd)) * psd) / np.sum(psd) 
            for psd in power_spectral_density.T  # Her kanalı ayrı işle
        ])

        # Özelliklerin birleştirilmesi
        return np.concatenate([
            mean_abs_value,
            root_mean_square,
            waveform_length,
            variance,
            integrated_emg,
            mean_frequency,
        ])


    def extract_features(self, data):
        features = []
        labels = []
        unique_classes = data["class"].unique()
        for gesture in tqdm(unique_classes, desc="Processing Classes"):
            subset = data[data["class"] == gesture]
            for i in tqdm(range(0, len(subset) - self.window_size, self.window_size), desc=f"Processing Class {gesture}", leave=False):
                window = subset.iloc[i:i + self.window_size][self.channels].values
                extracted_features = self.advanced_feature_extraction(window)
                features.append(extracted_features)
                labels.append(gesture)
        return np.array(features), np.array(labels)

    def train_model(self):
        # Load dataset
        data = pd.read_csv(self.file_path)

        # Data cleaning
        data = self.drop_columns(data, ["time", "label"])
        data = self.drop_na(data)
        data = self.drop_unmarked_class(data, "class", unmarked_value=0)

        # Filtering
        data = self.filter_all_channels(data)

        # Feature extraction
        features, labels = self.extract_features(data)

        # Train-test split
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # Balancing the data (SMOTE)
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Model training
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        # Validation metrics
        y_val_pred = model.predict(X_val)
        print("\nValidation Metrics:")
        print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"F1 Score: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
        print(f"Precision: {precision_score(y_val, y_val_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_val, y_val_pred, average='macro'):.4f}")

        # Test metrics
        y_test_pred = model.predict(X_test)
        print("\nTest Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_test_pred, average='macro'):.4f}")
        print(f"Precision: {precision_score(y_test, y_test_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_test, y_test_pred, average='macro'):.4f}")


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = LogRegTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=200, cutoff=(20, 450), sampling_rate=1000)
    trainer.train_model()
