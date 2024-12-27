import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import pywt

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

    # Özellik Çıkarımı
    def extract_features_from_window(self, window):
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

    def extract_from_data(self, data, labels):
        feature_matrix = []
        for i in range(data.shape[0]):
            window_features = self.extract_features_from_window(data[i, :])
            window_features["Label"] = labels[i]
            feature_matrix.append(window_features)
        return pd.DataFrame(feature_matrix)

if __name__ == "__main__":
    import os

    # Veri Yolu ve Çıktı Dizini
    data_path = "dataset/emg_data.csv"
    output_dir = "results/features"
    os.makedirs(output_dir, exist_ok=True)

    # Veri Yükleme
    print("Veri yükleniyor...")
    dataset = pd.read_csv(data_path)
    print("Veri başarıyla yüklendi.")

    # Özellikler ve Etiketler
    X = dataset.iloc[:, :-1].values  # Sensör Verileri
    y = dataset.iloc[:, -1].values  # Etiketler

    # Özellik Çıkarımı
    extractor = FeatureExtractor(window_size=8)
    print("Özellikler çıkarılıyor...")
    features_df = extractor.extract_from_data(X, y)

    # Sonuçların Kaydedilmesi
    output_file = os.path.join(output_dir, "feature_matrix_with_labels.csv")
    features_df.to_csv(output_file, index=False)
    print(f"Özellik matrisi ve etiketler '{output_file}' dosyasına kaydedildi.")
    print("Çıkarılan özelliklerin boyutu:", features_df.shape)

    print("\n--- Tüm İşlemler Tamamlandı ---")
