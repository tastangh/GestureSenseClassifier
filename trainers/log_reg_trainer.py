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

        # Log dosyasını ayarla
        self.log_file = os.path.join(self.save_dir, "log.txt")
        with open(self.log_file, "w") as f:
            f.write("Model Eğitim Logları\n")
            f.write("=" * 50 + "\n")

    def log(self, message):
        # Mesajı hem terminale yaz hem de log dosyasına kaydet
        print(message)
        with open(self.log_file, "a") as f:
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
        for channel in self.channels:
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

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)

        title = "Karışıklık Matrisi"
        feature_info = (
            f"Pencere Boyutu: {self.window_size}, "
            f"Kesim: {self.cutoff[0]}-{self.cutoff[1]} Hz, "
            f"Örnekleme Hızı: {self.sampling_rate} Hz"
        )
        ax.set_title(f"{title}\n{feature_info}", fontsize=12, pad=20)
        ax.set_xlabel("Tahmin Edilen", fontsize=10)
        ax.set_ylabel("Gerçek", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close(fig)

    def train_model(self):
        data = pd.read_csv(self.file_path)

        self.log("\nVeri Yükleme Tamamlandı")
        self.log(f"Toplam Veri Sayısı: {len(data)}")

        # Veri temizleme
        data = self.drop_columns(data, ["time", "label"])
        self.log("Sütunlar Silindi: ['time', 'label']")
        data = self.drop_na(data)
        self.log("Eksik Veriler Temizlendi")
        data = self.drop_unmarked_class(data, "class", unmarked_value=0)
        self.log("Sınıf 0 Temizlendi")

        self.log(f"\nSınıf Dağılımı (Temizleme Sonrası):\n{data['class'].value_counts()}")

        # Filtremleme
        data = self.filter_all_channels(data)
        self.log(f"Bant Geçiş Filtresi Uygulandı: {self.cutoff[0]}-{self.cutoff[1]} Hz")

        # Dengeleme
        data = self.balance_data_with_smote(data, class_column="class")
        self.log(f"Sınıf Dağılımı (Dengeleme Sonrası):\n{data['class'].value_counts()}")

        # Özellik çıkarımı
        features, labels = self.extract_features(data)
        self.log(f"\nÖzellik Çıkarımı Tamamlandı: {len(features)} örnek")

        # Eğitim, doğrulama ve test setlerine bölme
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        self.log(f"Eğitim, Doğrulama ve Test Setleri Ayrıldı")

        # Normalizasyon
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        self.log("Veri Normalizasyonu Tamamlandı")

        # Model eğitimi
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        self.log("Logistik Regresyon Modeli Eğitildi")

        # Performans değerlendirme
        metrics = []

        def evaluate_and_log(name, y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            metrics.append([name, accuracy, f1, precision, recall])
            self.log(f"\n{name} Metrikleri:\n"
                     f"Doğruluk: {accuracy:.4f}, F1 Skoru: {f1:.4f}, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}")

        # Doğrulama seti
        y_val_pred = model.predict(X_val)
        evaluate_and_log("Doğrulama", y_val, y_val_pred)
        self.plot_confusion_matrix(y_val, y_val_pred, classes=np.unique(labels), filename="dogrulama_karisiklik_matrisi.png")

        # Test seti
        y_test_pred = model.predict(X_test)
        evaluate_and_log("Test", y_test, y_test_pred)
        self.plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(labels), filename="test_karisiklik_matrisi.png")

        # Metrikleri tablo olarak kaydet
        metrics_df = pd.DataFrame(metrics, columns=["Set", "Doğruluk", "F1 Skoru", "Kesinlik", "Duyarlılık"])
        metrics_df.to_csv(os.path.join(self.save_dir, "metrikler.csv"), index=False)
        self.log("Tüm metrikler 'metrikler.csv' dosyasına kaydedildi.")


if __name__ == "__main__":
    channels = [f"channel{i}" for i in range(1, 9)]
    trainer = LogRegTrainer(file_path="dataset/EMG-data.csv", channels=channels, window_size=100, cutoff=(20, 450), sampling_rate=1000)
    trainer.train_model()
