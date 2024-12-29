import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import pywt
from scipy.signal import find_peaks


class EMGFeatureExtractor:
    def __init__(self, data, channels, window_size=200, sampling_rate=1000):
        """
        EMG Feature Extraction Class.
        :param data: DataFrame containing EMG signals.
        :param channels: List of channel names in the DataFrame.
        :param window_size: Number of samples in each segment.
        :param sampling_rate: Sampling rate of the EMG signal.
        """
        self.data = data
        self.channels = channels
        self.window_size = window_size
        self.sampling_rate = sampling_rate

    @staticmethod
    def compute_time_domain_features(signal):
        """
        Compute time-domain features.
        :param signal: 1D array of EMG signal.
        :return: Dictionary of time-domain features.
        """
        features = {
            "MAV": np.mean(np.abs(signal)),  # Mean Absolute Value
            "RMS": np.sqrt(np.mean(signal ** 2)),  # Root Mean Square
            "WL": np.sum(np.abs(np.diff(signal))),  # Waveform Length
            "Variance": np.var(signal),  # Variance
            "Skewness": skew(signal),  # Skewness
            "Kurtosis": kurtosis(signal),  # Kurtosis
        }

        # Zero Crossing
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features["Zero_Crossing"] = zero_crossings

        # Slope Sign Change
        slope_changes = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
        features["Slope_Sign_Change"] = slope_changes

        # Peak Amplitude
        peaks, _ = find_peaks(signal)
        features["Peak_Amplitude"] = signal[peaks].max() if len(peaks) > 0 else 0

        return features

    @staticmethod
    def compute_frequency_domain_features(signal, sampling_rate=1000):
        """
        Compute frequency-domain features.
        :param signal: 1D array of EMG signal.
        :param sampling_rate: Sampling rate of the EMG signal.
        :return: Dictionary of frequency-domain features.
        """
        freqs, psd = welch(signal, fs=sampling_rate)
        total_power = np.sum(psd)
        mean_freq = np.sum(freqs * psd) / total_power
        median_freq = freqs[np.cumsum(psd) >= total_power / 2][0]

        features = {
            "Mean_Frequency": mean_freq,
            "Median_Frequency": median_freq,
            "Total_Power": total_power,
        }
        return features

    @staticmethod
    def compute_wavelet_features(signal, wavelet="db4", level=4):
        """
        Compute wavelet-based features.
        :param signal: 1D array of EMG signal.
        :param wavelet: Wavelet type.
        :param level: Number of decomposition levels.
        :return: Dictionary of wavelet features.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        wavelet_energy = [np.sum(c ** 2) for c in coeffs]

        features = {f"Wavelet_Energy_L{idx}": energy for idx, energy in enumerate(wavelet_energy)}
        return features

    @staticmethod
    def compute_entropy_features(signal):
        """
        Compute entropy-based features.
        :param signal: 1D array of EMG signal.
        :return: Dictionary of entropy features.
        """
        # Approximate Entropy
        approx_entropy = np.sum(np.abs(np.diff(signal))) / len(signal)
        # Sample Entropy
        sample_entropy = -np.sum(signal * np.log2(signal + 1e-10))

        features = {
            "Approximate_Entropy": approx_entropy,
            "Sample_Entropy": sample_entropy,
        }
        return features

    @staticmethod
    def compute_hjorth_parameters(signal):
        """
        Compute Hjorth parameters.
        :param signal: 1D array of EMG signal.
        :return: Dictionary of Hjorth parameters.
        """
        diff_signal = np.diff(signal)
        var_zero = np.var(signal)
        var_d1 = np.var(diff_signal)
        diff_d1 = np.diff(diff_signal)
        var_d2 = np.var(diff_d1)

        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / mobility

        return {
            "Hjorth_Activity": activity,
            "Hjorth_Mobility": mobility,
            "Hjorth_Complexity": complexity,
        }

    def extract_features(self):
        """
        Extract features from all channels and all segments.
        :return: DataFrame with features for each channel and segment.
        """
        feature_list = []
        num_samples = len(self.data)

        # Iterate over segments
        for start in range(0, num_samples, self.window_size):
            end = start + self.window_size
            if end > num_samples:
                break

            segment = self.data.iloc[start:end]

            # Extract features for the segment
            segment_features = {"Segment_Start": start, "Segment_End": end}
            for channel in self.channels:
                signal = segment[channel].values

                # Time-domain features
                time_features = self.compute_time_domain_features(signal)

                # Frequency-domain features
                freq_features = self.compute_frequency_domain_features(signal, self.sampling_rate)

                # Wavelet features
                wavelet_features = self.compute_wavelet_features(signal)

                # Entropy features
                entropy_features = self.compute_entropy_features(signal)

                # Hjorth parameters
                hjorth_features = self.compute_hjorth_parameters(signal)

                # Combine all features for the current channel
                channel_features = {
                    f"{channel}_{key}": value
                    for key, value in {**time_features, **freq_features, **wavelet_features, **entropy_features, **hjorth_features}.items()
                }
                segment_features.update(channel_features)

            # Add the segment's class label
            segment_features["class"] = segment["class"].iloc[0]
            feature_list.append(segment_features)

        return pd.DataFrame(feature_list)


if __name__ == "__main__":
    # Veri Yükleme
    file_path = "./dataset/filtered_emg_data.csv"
    data = pd.read_csv(file_path)

    # Kanal İsimleri
    channels = [f"channel{i}" for i in range(1, 9)]

    # EMGFeatureExtractor Sınıfını Başlat
    extractor = EMGFeatureExtractor(data, channels, window_size=200, sampling_rate=1000)

    # Özellik Çıkarımı
    features = extractor.extract_features()

    # Özellikleri Kaydetme
    output_path = "./dataset/emg_features.csv"
    features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
