import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import pywt

class FeatureExtractor:
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
        """Integrated EMG (IEMG): Sinyalin toplam genliği."""
        return np.sum(np.abs(signal))

    def damv(self, signal):
        """Difference Absolute Mean Value (DAMV)."""
        return np.mean(np.abs(np.diff(signal)))

    def hjorth_parameters(self, signal):
        """Hjorth Parametreleri."""
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
            self.peak_to_peak(signal)
        ]

    def peak_to_peak(self, signal):
        return np.ptp(signal)

    # Wavelet Özellikleri
    def wavelet_energy(self, signal):
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        return [np.sum(c ** 2) for c in coeffs]

    # Özellik Çıkarımı
    def extract_features_from_window(self, window):
        features = []
        # Zaman Alanı
        features.append(self.mean_absolute_value(window))
        features.append(self.root_mean_square(window))
        features.append(self.waveform_length(window))
        features.append(self.zero_crossings(window))
        features.append(self.slope_sign_changes(window))
        features.append(self.log_detector(window))
        features.append(self.integrated_emg(window))
        features.append(self.damv(window))
        activity, mobility, complexity = self.hjorth_parameters(window)
        features.extend([activity, mobility, complexity])
        # Frekans Alanı
        features.append(self.mean_frequency(window))
        features.append(self.median_frequency(window))
        features.append(self.spectral_entropy(window))
        features.append(self.spectral_energy(window))
        # İstatistiksel
        features.extend(self.statistical_features(window))
        # Wavelet
        features.extend(self.wavelet_energy(window))
        return features

    def extract_from_data(self, data):
        num_samples, num_sensors = data.shape
        feature_matrix = []

        for sensor in range(num_sensors):
            sensor_data = data[:, sensor]
            for i in range(0, len(sensor_data), self.window_size):
                window = sensor_data[i:i + self.window_size]
                if len(window) == self.window_size:
                    features = self.extract_features_from_window(window)
                    feature_matrix.append(features)

        return np.array(feature_matrix)
