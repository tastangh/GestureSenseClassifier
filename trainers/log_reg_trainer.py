import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class LogRegTrainer:
    def __init__(self, file_path, channels, window_size=200, cutoff=(20, 450), sampling_rate=1000, base_dir="results"):
        self.file_path = file_path
        self.channels = channels
        self.window_size = window_size
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate

        # Parametrelere dayalı alt klasör oluşturma
        param_dir = f"log_reg_ws{window_size}_cutoff{cutoff[0]}-{cutoff[1]}_sr{sampling_rate}"
        self.save_dir = os.path.join(base_dir, param_dir)
        os.makedirs(self.save_dir, exist_ok=True)

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
    def balance_data_with_smote(data, class_column):
        print("\nClass Counts Before Balancing:")
        print(data[class_column].value_counts())

        smote = SMOTE(random_state=42)
        X = data.drop(columns=[class_column]).values
        y = data[class_column].values
        X_balanced, y_balanced = smote.fit_resample(X, y)

        balanced_data = pd.DataFrame(X_balanced, columns=data.columns[:-1])
        balanced_data[class_column] = y_balanced

        print("\nClass Counts After Balancing:")
        print(pd.Series(y_balanced).value_counts())
        print(f"\nTotal Data After Balancing: {len(balanced_data)} samples")
        return balanced_data

    @staticmethod
    def advanced_feature_extraction(window):
        mean_abs_value = np.mean(np.abs(window), axis=0)
        root_mean_square = np.sqrt(np.mean(np.square(window), axis=0))
        waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        variance = np.var(window, axis=0)
        integrated_emg = np.sum(np.abs(window), axis=0)

        freq_spectrum = np.fft.fft(window, axis=0)
        power_spectral_density = np.abs(freq_spectrum) ** 2
        mean_frequency = np.array([
            np.sum(np.arange(len(psd)) * psd) / np.sum(psd) 
            for psd in power_spectral_density.T
        ])

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

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)

        # Başlık ve bilgilerin eklenmesi
        title = "Confusion Matrix"
        feature_info = (
            f"Window Size: {self.window_size}, "
            f"Cutoff: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Sampling Rate: {self.sampling_rate} Hz\n"
            f"Features: Mean, RMS, WL, Variance, Integrated EMG, Mean Frequency"
        )
        ax.set_title(f"{title}\n{feature_info}", fontsize=12, pad=20)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)

        # Kaydedilmesi
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close(fig)


    def train_model(self):
        data = pd.read_csv(self.file_path)

        # Data cleaning
        data = self.drop_columns(data, ["time", "label"])
        data = self.drop_na(data)
        data = self.drop_unmarked_class(data, "class", unmarked_value=0)

        print("\nClass Counts Before Processing:")
        print(data["class"].value_counts())

        # Filtering
        data = self.filter_all_channels(data)

        # Balancing data
        data = self.balance_data_with_smote(data, class_column="class")
        print(data["class"].value_counts())

        # Feature extraction
        features, labels = self.extract_features(data)

        print(f"\nData After Feature Extraction: {len(features)} samples, {len(set(labels))} classes")

        # Train-test split
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

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
        self.plot_confusion_matrix(y_val, y_val_pred, classes=np.unique(labels), filename="validation_confusion_matrix.png")

        # Test metrics
        y_test_pred = model.predict(X_test)
        print("\nTest Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_test_pred, average='macro'):.4f}")
        print(f"Precision: {precision_score(y_test, y_test_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_test, y_test_pred, average='macro'):.4f}")
        self.plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(labels), filename="test_confusion_matrix.png")


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = LogRegTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=100, cutoff=(20, 450), sampling_rate=1000)
    trainer.train_model()
