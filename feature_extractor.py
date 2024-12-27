import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import os


class FeatureExtractor:
    """
    EMG verilerinden özellik çıkarımı yapan sınıf.
    """
    def __init__(self, window_size=8):
        self.window_size = window_size

    # Zaman Alanı Özellikleri
    def mean_absolute_value(self, signal):
        return np.mean(np.abs(signal))

    def root_mean_square(self, signal):
        return np.sqrt(np.mean(signal ** 2))

    def waveform_length(self, signal):
        return np.sum(np.abs(np.diff(signal)))

    def zero_crossings(self, signal, threshold=0.01):
        return np.sum(np.abs(np.diff(np.sign(signal))) > threshold)

    def slope_sign_changes(self, signal, threshold=0.01):
        return np.sum(np.diff(np.sign(np.diff(signal))) > threshold)

    def log_detector(self, signal):
        return np.mean(np.log1p(np.abs(signal)))

    def integrated_emg(self, signal):
        return np.sum(np.abs(signal))

    def damv(self, signal):
        return np.mean(np.abs(np.diff(signal)))

    def variance(self, signal):
        return np.var(signal)

    def enhanced_mav(self, signal, threshold=0.01):
        return np.mean([abs(x) for x in signal if abs(x) > threshold])

    def enhanced_wl(self, signal, threshold=0.01):
        return np.sum([abs(signal[i] - signal[i - 1]) for i in range(1, len(signal)) if abs(signal[i] - signal[i - 1]) > threshold])

    def myopulse_percentage_rate(self, signal, threshold=0.01):
        return np.sum(np.abs(signal) > threshold) / len(signal)

    def simple_square_integral(self, signal):
        return np.sum(signal ** 2)

    def willison_amplitude(self, signal, threshold=0.01):
        return np.sum(np.abs(np.diff(signal)) > threshold)

    def hjorth_parameters(self, signal):
        first_derivative = np.diff(signal)
        second_derivative = np.diff(first_derivative)
        activity = np.var(signal)
        mobility = np.sqrt(np.var(first_derivative) / activity)
        complexity = np.sqrt(np.var(second_derivative) / np.var(first_derivative))
        return activity, mobility, complexity

    # Frekans Alanı Özellikleri
    def mean_frequency(self, signal):
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), d=1/200)
        return np.sum(freqs * fft_vals) / np.sum(fft_vals)

    def median_frequency(self, signal):
        fft_vals = np.abs(fft(signal))
        cumsum = np.cumsum(fft_vals)
        return np.argmax(cumsum >= np.sum(fft_vals) / 2)

    def spectral_energy(self, signal):
        fft_vals = np.abs(fft(signal)) ** 2
        return np.sum(fft_vals)

    def spectral_entropy(self, signal):
        fft_vals = np.abs(fft(signal))
        fft_power = fft_vals ** 2
        power_sum = np.sum(fft_power)
        prob_power = fft_power / power_sum
        return -np.sum(prob_power * np.log2(prob_power + 1e-10))

    # İstatistiksel Özellikler
    def statistical_features(self, signal):
        return [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            skew(signal),
            kurtosis(signal),
            np.ptp(signal)
        ]

    # Wavelet Özellikleri
    def wavelet_energy(self, signal):
        coeffs = pywt.wavedec(signal, 'db4', level=min(4, pywt.dwt_max_level(len(signal), pywt.Wavelet('db4'))))
        return [np.sum(c ** 2) for c in coeffs]

    def extract_features(self, dataset):
        """
        Verilen veri setinden özellik çıkarır.
        :param dataset: DataFrame formatında veri seti (son sütun etiketleri içerir)
        :return: Özellik matrisi (DataFrame)
        """
        if dataset is None or dataset.empty:
            raise ValueError("Boş bir veri seti sağlandı.")

        X = dataset.iloc[:, :-1].values  # Sensör verileri
        y = dataset.iloc[:, -1].values  # Etiketler

        feature_matrix = []
        for i in range(X.shape[0]):
            window_features = self.extract_features_from_window(X[i, :])
            window_features["Label"] = y[i]
            feature_matrix.append(window_features)

        return pd.DataFrame(feature_matrix)

    def extract_features_from_window(self, window):
        """
        Tek bir pencere için özellik çıkarımı.
        """
        features = {}
        # Zaman Alanı
        features["MAV"] = self.mean_absolute_value(window)
        features["RMS"] = self.root_mean_square(window)
        features["WaveformLength"] = self.waveform_length(window)
        features["ZeroCrossings"] = self.zero_crossings(window)
        features["SlopeSignChanges"] = self.slope_sign_changes(window)
        features["LogDetector"] = self.log_detector(window)
        features["IEMG"] = self.integrated_emg(window)
        features["DAMV"] = self.damv(window)
        features["Variance"] = self.variance(window)
        features["EnhancedMAV"] = self.enhanced_mav(window)
        features["EnhancedWL"] = self.enhanced_wl(window)
        features["MYOP"] = self.myopulse_percentage_rate(window)
        features["SSI"] = self.simple_square_integral(window)
        features["WAMP"] = self.willison_amplitude(window)
        activity, mobility, complexity = self.hjorth_parameters(window)
        features["HjorthActivity"] = activity
        features["HjorthMobility"] = mobility
        features["HjorthComplexity"] = complexity
        # Frekans Alanı
        features["MeanFrequency"] = self.mean_frequency(window)
        features["MedianFrequency"] = self.median_frequency(window)
        features["SpectralEntropy"] = self.spectral_entropy(window)
        features["SpectralEnergy"] = self.spectral_energy(window)
        # İstatistiksel
        stats = self.statistical_features(window)
        stat_names = ["Mean", "StdDev", "Min", "Max", "Skewness", "Kurtosis", "PeakToPeak"]
        features.update({name: value for name, value in zip(stat_names, stats)})
        # Wavelet
        wavelet_features = self.wavelet_energy(window)
        wavelet_names = [f"WaveletEnergy_Level{i+1}" for i in range(len(wavelet_features))]
        features.update({name: value for name, value in zip(wavelet_names, wavelet_features)})
        return features

    def plot_feature_distributions_by_class_combined(self, features_df, output_dir="results/features/plots", title="Feature Distributions by Class"):
        """
        Özelliklerin sınıflara göre dağılımını tek bir grafikte gösterir ve kaydeder.
        :param features_df: Özellik matrisi (DataFrame)
        :param output_dir: Grafiklerin kaydedileceği dizin
        :param title: Grafik başlığı
        """
        os.makedirs(output_dir, exist_ok=True)
        labels = features_df["Label"].unique()
        num_features = len(features_df.columns) - 1  # 'Label' hariç sütunlar
        num_rows = int(np.ceil(num_features / 5))  # 5 sütunlu grid yapısı

        plt.figure(figsize=(20, 4 * num_rows))
        for i, feature in enumerate(features_df.columns[:-1]):  # 'Label' hariç özellikler
            plt.subplot(num_rows, 5, i + 1)
            for label in labels:
                sns.kdeplot(
                    features_df[features_df["Label"] == label][feature],
                    label=f"Class {int(label)}",
                    fill=True,
                    alpha=0.5
                )
            plt.title(feature, fontsize=8)
            plt.legend(loc="upper right", fontsize=6)
            plt.tight_layout()

        save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.savefig(save_path)
        plt.close()
        print(f"Tüm sınıflar için özellik dağılımı görselleştirme kaydedildi: {save_path}")


    def plot_feature_comparison_by_class_combined(self, features_df1, features_df2, output_dir="results/features/plots", source1="Unfiltered", source2="Filtered", title="Feature Comparison by Class"):
        """
        Filtrelenmiş ve filtrelenmemiş verilerin sınıflara göre özellik karşılaştırmalarını tek bir grafikte gösterir.
        :param features_df1: Birinci özellik matrisi
        :param features_df2: İkinci özellik matrisi
        :param output_dir: Grafiklerin kaydedileceği dizin
        :param source1: Birinci veri kümesinin adı
        :param source2: İkinci veri kümesinin adı
        :param title: Grafik başlığı
        """
        os.makedirs(output_dir, exist_ok=True)
        labels = features_df1["Label"].unique()
        num_features = len(features_df1.columns) - 1  # 'Label' hariç sütunlar
        num_rows = int(np.ceil(num_features / 5))  # 5 sütunlu grid yapısı

        plt.figure(figsize=(20, 4 * num_rows))
        for i, feature in enumerate(features_df1.columns[:-1]):  # 'Label' hariç özellikler
            plt.subplot(num_rows, 5, i + 1)
            for label in labels:
                sns.kdeplot(
                    features_df1[features_df1["Label"] == label][feature],
                    label=f"{source1} - Class {int(label)}",
                    fill=True,
                    alpha=0.3
                )
                sns.kdeplot(
                    features_df2[features_df2["Label"] == label][feature],
                    label=f"{source2} - Class {int(label)}",
                    fill=True,
                    alpha=0.3
                )
            plt.title(feature, fontsize=8)
            plt.legend(loc="upper right", fontsize=6)
            plt.tight_layout()

        save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.savefig(save_path)
        plt.close()
        print(f"Tüm sınıflar için özellik karşılaştırma görselleştirme kaydedildi: {save_path}")

if __name__ == "__main__":
    from data_processor import DataProcessor

    # Veri Yolları
    raw_data_path = "dataset/emg_data.csv"
    filtered_data_path = "filtered_dataset/emg_filtered_data.csv"

    # DataProcessor Nesnelerini Kullanarak Verileri Yükle
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    raw_processor = DataProcessor(class_names)
    raw_processor.set_data_path(raw_data_path)
    raw_dataset = raw_processor.load_data()

    filtered_processor = DataProcessor(class_names)
    filtered_processor.set_data_path(filtered_data_path)
    filtered_dataset = filtered_processor.load_data()

    # FeatureExtractor Nesnesini Kullanarak Özellik Çıkar
    extractor = FeatureExtractor(window_size=8)

    print("Filtrelenmemiş veri için özellikler çıkarılıyor...")
    raw_features_df = extractor.extract_features(raw_dataset)

    print("Filtrelenmiş veri için özellikler çıkarılıyor...")
    filtered_features_df = extractor.extract_features(filtered_dataset)

    # Özellikleri Kaydetme
    feature_output_dir = "results/features"
    os.makedirs(feature_output_dir, exist_ok=True)

    raw_features_file = os.path.join(feature_output_dir, "raw_features_with_labels.csv")
    raw_features_df.to_csv(raw_features_file, index=False)
    print(f"Filtrelenmemiş özellikler kaydedildi: {raw_features_file}")

    filtered_features_file = os.path.join(feature_output_dir, "filtered_features_with_labels.csv")
    filtered_features_df.to_csv(filtered_features_file, index=False)
    print(f"Filtrelenmiş özellikler kaydedildi: {filtered_features_file}")

    # Görselleştirme
    print("Tüm sınıflar için filtrelenmemiş özellik dağılımı görselleştiriliyor...")
    extractor.plot_feature_distributions_by_class_combined(raw_features_df, output_dir="results/features/plots", title="Unfiltered Feature Distributions by Class")

    print("Tüm sınıflar için filtrelenmiş özellik dağılımı görselleştiriliyor...")
    extractor.plot_feature_distributions_by_class_combined(filtered_features_df, output_dir="results/features/plots", title="Filtered Feature Distributions by Class")

    print("Filtrelenmiş ve filtrelenmemiş verilerden çıkarılan özellikler sınıf bazında karşılaştırılıyor...")
    extractor.plot_feature_comparison_by_class_combined(
        raw_features_df, 
        filtered_features_df, 
        output_dir="results/features/plots", 
        source1="Unfiltered", 
        source2="Filtered", 
        title="Feature Comparison by Class"
    )
