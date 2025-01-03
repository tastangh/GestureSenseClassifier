import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class FeatureExtractor:
    def __init__(self, data, window_size=200, overlap=0, selected_features=None, feature_weights=None):
        """
        FeatureExtractor sınıfı, belirtilen özellikleri bir kayar pencere üzerinden çıkarır.
        :param data: Veri seti (pandas DataFrame)
        :param window_size: Kayar pencere boyutu
        :param overlap: Pencereler arası örtüşme oranı (0-1 arası)
        :param selected_features: Çıkarılacak özelliklerin listesi
        :param feature_weights: Özellik ağırlıkları (dict: {feature_name: weight})
        """
        self.data = data
        self.window_size = window_size
        self.overlap = overlap
        self.selected_features = selected_features
        self.feature_weights = feature_weights or {}

        # Özellik isimlerini kısaltmalarıyla tanımlama
        self.feature_abbreviations = {
            "mean": "Mean",
            "std": "STD",
            "min": "Min",
            "max": "Max",
            "energy": "Energy",
            "zero_crossing_rate": "ZCR",
            "dominant_frequency": "DF",
            "rms": "RMS",
            "waveform_length": "WL",
            "mean_absolute_value": "MAV",
             "variance": "VAR",
            "slope_sign_changes": "SSC",
            "willison_amplitude": "WAMP",
            "myopulse_percentage_rate": "MPR",
            "simple_square_integral": "SSI",
            "log_detector": "LD",
            "difference_absolute_standard_deviation_value": "DASDV",
            "maximum_fractal_length": "MFL",
        }

         # Tüm özellikler
        self.available_features = {
            "mean": lambda window: np.mean(window, axis=0),
            "std": lambda window: np.std(window, axis=0),
            "min": lambda window: np.min(window, axis=0),
            "max": lambda window: np.max(window, axis=0),
            "energy": lambda window: np.sum(np.square(window), axis=0),
            "zero_crossing_rate": lambda window: np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0),
            "dominant_frequency": lambda window: np.argmax(np.abs(np.fft.fft(window, axis=0)), axis=0)[0] if window.size > 0 else 0,
             "rms": lambda window: np.sqrt(np.mean(np.square(window), axis=0)),
            "waveform_length": lambda window: np.sum(np.abs(np.diff(window, axis=0)), axis=0),
            "mean_absolute_value": lambda window: np.mean(np.abs(window), axis=0),
            "variance": lambda window: np.var(window, axis=0),
           "slope_sign_changes": lambda window: np.mean(np.sum(np.diff(np.sign(np.diff(window, axis=0)), axis=0) != 0, axis=0)) if window.size > 0 else 0,
             "willison_amplitude": lambda window: np.mean(np.sum(np.abs(np.diff(window, axis=0)) > 0.01, axis=0)) if window.size > 0 else 0,
            "myopulse_percentage_rate": lambda window: np.sum(np.abs(window) > 0.01, axis=0) / window.shape[0],
            "simple_square_integral": lambda window: np.sum(window**2, axis=0),
            "log_detector": lambda window: np.exp(np.mean(np.log(np.abs(window) + 1e-10), axis=0)),
            "difference_absolute_standard_deviation_value": lambda window: np.mean(np.abs(np.diff(window, axis=0)), axis=0),
             "maximum_fractal_length": lambda window: np.sum(np.abs(np.diff(window, axis=0))**1.5, axis=0),
         }

    def extract_features(self, data=None):
        """
        Seçilen özellikleri çıkarır ve bir pandas DataFrame olarak döner.
        :param data: Veri seti (pandas DataFrame)
        :return: Çıkarılan özellikler (pandas DataFrame)
        """
        if data is not None:
            self.data = data

        step_size = int(self.window_size * (1 - self.overlap))
        windows = [
            self.data.iloc[start:start + self.window_size].values
            for start in range(0, len(self.data) - self.window_size + 1, step_size)
        ]

        feature_matrix = []

        for window in windows:
            feature_row = {}
            for feature_name, func in self.available_features.items():
                if self.selected_features is None or feature_name in self.selected_features:
                    feature_values = func(window)
                    weight = self.feature_weights.get(feature_name, 1.0)
                    feature_row[feature_name] = feature_values * weight
            feature_matrix.append(feature_row)

        return pd.DataFrame(feature_matrix)

    def save_feature_importance_plot(self, output_dir):
        """
        Özellik ağırlıklarını çizer ve kaydeder.
        :param output_dir: Grafiklerin kaydedileceği klasör
        """
        os.makedirs(output_dir, exist_ok=True)

        if not self.feature_weights:
            print("Ağırlıklandırma bilgisi bulunamadı.")
            return

        # Kısaltılmış isimler
        shortened_labels = [self.feature_abbreviations.get(k, k) for k in self.feature_weights.keys()]

        plt.figure(figsize=(10, 6))
        plt.bar(shortened_labels, self.feature_weights.values(), color="skyblue")
        plt.title("Özellik Ağırlıkları")
        plt.xlabel("Özellikler")
        plt.ylabel("Ağırlık")
        plt.xticks(rotation=45)
        plt.grid(axis='y')

        plot_path = os.path.join(output_dir, "feature_weights.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Özellik ağırlık grafiği kaydedildi: {plot_path}")


if __name__ == "__main__":
    # Veri seti
    dataset_path = "dataset/filtered_balanced_EMG_data.csv"
    data = pd.read_csv(dataset_path)

    # Kanal isimleri
    channels = [f"channel{i}" for i in range(1, 9)]

   # Özellik çıkarma sınıfını başlat
    selected_features = [
        "mean_absolute_value", "rms", "waveform_length",
        "slope_sign_changes", "willison_amplitude", "dominant_frequency",
         "maximum_fractal_length", "log_detector"
    ]

    # Literatür tabanlı ağırlıklandırma
    feature_weights = {
        "mean_absolute_value": 1.5,
        "rms": 1.5,
        "waveform_length": 1.3,
        "slope_sign_changes": 1.2,
        "willison_amplitude": 1.2,
        "dominant_frequency": 1.0,
        "maximum_fractal_length": 0.8,
        "log_detector": 0.7
    }


    # FeatureExtractor örneği oluştur
    extractor = FeatureExtractor(
        data[channels],
        window_size=200,
        overlap=0.5,
        selected_features=selected_features,
        feature_weights=feature_weights
    )

    # Özellikleri çıkar
    print("Özellikler çıkarılıyor...")
    features = extractor.extract_features()
    print("Özellikler başarıyla çıkarıldı:")
    print(features.head())

    # Özellik ağırlıklarını kaydet
    print("\nÖzellik Ağırlıkları Görselleştiriliyor ve Kaydediliyor...")
    output_dir = "plots"
    extractor.save_feature_importance_plot(output_dir)

    # Özellikleri kaydet
    output_path = "dataset/extracted_features_filtered_balanced.csv"
    features.to_csv(output_path, index=False)
    print(f"Çıkarılan özellikler kaydedildi: {output_path}")