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

        # Ana klasör ayarı
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def log(self, log_dir, message):
        """Hem terminale yaz hem de log dosyasına kaydet."""
        print(message)
        log_file = os.path.join(log_dir, "log.txt")
        with open(log_file, "a") as f:
            f.write(message + "\n")

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
            raise ValueError("Sadece bant geçiş filtresi destekleniyor.")
        return filtfilt(b, a, signal)

    def filter_all_channels(self, data):
        for channel in tqdm(self.channels, desc="Kanallara Filtre Uygulanıyor"):
            data[channel] = self.apply_filter(data[channel].values, "band", self.cutoff, order=4, sampling_rate=self.sampling_rate)
        return data

    @staticmethod
    def balance_data_with_smote(data, class_column):
        smote = SMOTE(random_state=42)
        X = data.drop(columns=[class_column]).values
        y = data[class_column].values
        X_balanced, y_balanced = smote.fit_resample(X, y)

        balanced_data = pd.DataFrame(X_balanced, columns=data.columns[:-1])
        balanced_data[class_column] = y_balanced
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
        for gesture in tqdm(unique_classes, desc="Sınıflar İşleniyor"):
            subset = data[data["class"] == gesture]
            for i in tqdm(range(0, len(subset) - self.window_size, self.window_size), desc=f"Sınıf {gesture} İşleniyor", leave=False):
                window = subset.iloc[i:i + self.window_size][self.channels].values
                extracted_features = self.advanced_feature_extraction(window)
                features.append(extracted_features)
                labels.append(gesture)
        return np.array(features), np.array(labels)

    def plot_confusion_matrix(self, log_dir, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)

        ax.set_title("Karışıklık Matrisi", fontsize=14)
        ax.set_xlabel("Tahmin Edilen", fontsize=12)
        ax.set_ylabel("Gerçek", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, filename))
        plt.close(fig)

    def save_metrics(self, log_dir, set_name, y_true, y_pred):
        """Başarı metriklerini CSV'ye kaydet."""
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        metrics_file = os.path.join(log_dir, "metrikler.csv")

        # Metrikleri dosyaya ekle
        with open(metrics_file, "a") as f:
            if os.stat(metrics_file).st_size == 0:
                f.write("Set,Doğruluk,F1 Skoru,Kesinlik,Duyarlılık\n")
            f.write(f"{set_name},{accuracy:.4f},{f1:.4f},{precision:.4f},{recall:.4f}\n")

    def train_scenario(self, scenario_name, data, include_features=True, include_normalization=True):
        scenario_dir = os.path.join(self.base_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        self.log(scenario_dir, f"\nSenaryo: {scenario_name} başlatıldı.")

        # Özellik çıkarımı
        if include_features:
            features, labels = self.extract_features(data)
            self.log(scenario_dir, "Özellik çıkarımı tamamlandı.")
        else:
            features = data.drop(columns=["class"]).values
            labels = data["class"].values
            self.log(scenario_dir, "Özellik çıkarımı atlandı.")

        # Eğitim, doğrulama ve test setlerine bölme
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        self.log(scenario_dir, "Veri eğitim, doğrulama ve test setlerine ayrıldı.")

        # Normalizasyon
        if include_normalization:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            self.log(scenario_dir, "Veri normalizasyonu tamamlandı.")
        else:
            self.log(scenario_dir, "Normalizasyon atlandı.")

        # Model eğitimi
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        self.log(scenario_dir, "Logistik Regresyon modeli eğitildi.")

        # Performans değerlendirme
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Karışıklık matrislerini kaydet
        self.plot_confusion_matrix(scenario_dir, y_val, y_val_pred, np.unique(labels), "dogrulama_karisiklik_matrisi.png")
        self.plot_confusion_matrix(scenario_dir, y_test, y_test_pred, np.unique(labels), "test_karisiklik_matrisi.png")
        self.log(scenario_dir, "Karışıklık matrisleri kaydedildi.")

        # Başarı metriklerini kaydet
        self.save_metrics(scenario_dir, "Doğrulama", y_val, y_val_pred)
        self.save_metrics(scenario_dir, "Test", y_test, y_test_pred)

    def analyze_processing_steps(self):
        data = pd.read_csv(self.file_path)

        self.log(self.base_dir, f"Veri yüklendi. Toplam örnek sayısı: {len(data)}")
        self.train_scenario("orijinal", data, include_features=False, include_normalization=False)

        # Veri temizleme
        data_cleaned = self.drop_na(data)
        self.train_scenario("temizlenmis", data_cleaned, include_features=False, include_normalization=False)

        # Sınıf 0'ı silme
        data_no_class0 = self.drop_unmarked_class(data_cleaned, "class", unmarked_value=0)
        self.train_scenario("sinif0_silindi", data_no_class0, include_features=False, include_normalization=False)

        # Filtremleme
        data_filtered = self.filter_all_channels(data_no_class0)
        self.train_scenario("filtrelendi", data_filtered, include_features=False, include_normalization=False)

        # Dengeleme
        data_balanced = self.balance_data_with_smote(data_filtered, "class")
        self.train_scenario("dengelendi", data_balanced, include_features=False, include_normalization=False)

        # Özellik çıkarımı
        features, labels = self.extract_features(data_balanced)
        features_df = pd.DataFrame(features)
        features_df["class"] = labels
        self.train_scenario("ozellik_cikarildi", features_df, include_features=False, include_normalization=True)


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = LogRegTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=100, cutoff=(20, 450), sampling_rate=1000)
    trainer.analyze_processing_steps()
