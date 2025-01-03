import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch


class FeatureExtractor:
    def __init__(self, data, channels, target_column):
        """
        EMG verilerinden özellik çıkarmak için sınıf.
        :param data: Veri seti (pandas DataFrame)
        :param channels: EMG kanalları (örneğin ['channel1', 'channel2', ...])
        :param target_column: Hedef sınıf sütunu
        """
        self.data = data
        self.channels = channels
        self.target_column = target_column

    def extract_features(self):
        """
        Zaman ve frekans alanı özelliklerini çıkarır.
        :return: Özellik çıkarılmış DataFrame
        """
        features = []

        for channel in self.channels:
            channel_data = self.data[channel]
            features.append(self._time_domain_features(channel_data, channel))
            features.append(self._frequency_domain_features(channel_data, channel))

        # Özellikleri birleştir
        feature_df = pd.concat(features, axis=1)
        feature_df[self.target_column] = self.data[self.target_column]

        return feature_df

    def _time_domain_features(self, signal, channel):
        """
        Zaman alanı özelliklerini hesaplar.
        :param signal: Sinyal verisi (numpy array veya pandas Series)
        :param channel: Kanal adı
        :return: Zaman alanı özellikleri DataFrame
        """
        return pd.DataFrame({
            f"{channel}_mean": [np.mean(signal)],
            f"{channel}_std": [np.std(signal)],
            f"{channel}_max": [np.max(signal)],
            f"{channel}_min": [np.min(signal)],
            f"{channel}_rms": [np.sqrt(np.mean(signal**2))],
            f"{channel}_skewness": [skew(signal)],
            f"{channel}_kurtosis": [kurtosis(signal)],
        })

    def _frequency_domain_features(self, signal, channel):
        """
        Frekans alanı özelliklerini hesaplar.
        :param signal: Sinyal verisi (numpy array veya pandas Series)
        :param channel: Kanal adı
        :return: Frekans alanı özellikleri DataFrame
        """
        freqs, psd = welch(signal, fs=1000)
        total_power = np.sum(psd)
        return pd.DataFrame({
            f"{channel}_mean_freq": [np.sum(freqs * psd) / total_power],
            f"{channel}_median_freq": [np.median(freqs)],
            f"{channel}_total_power": [total_power],
        })
