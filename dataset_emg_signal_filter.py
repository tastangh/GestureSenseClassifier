import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


class EMGSignalProcessor:
    def __init__(self, data, channels):
        """
        EMGSignalProcessor sınıfını başlatır.
        :param data: EMG veri seti (pandas DataFrame)
        :param channels: EMG kanallarının isimleri (örneğin, ['channel1', 'channel2', ...])
        """
        self.data = data
        self.channels = channels
        self.filtered_data = None  # Filtrelenmiş veriyi saklar

    def apply_filter(self, channel_name, filter_type, cutoff, order=4, sampling_rate=1000):
        """
        EMG sinyaline filtre uygular.
        :param channel_name: Filtre uygulanacak kanalın adı
        :param filter_type: 'low', 'high', veya 'band'
        :param cutoff: Kesim frekansı (low/high için tek değer, band için [low, high])
        :param order: Filtre düzeni
        :param sampling_rate: Sinyal örnekleme frekansı
        :return: Filtrelenmiş sinyal
        """
        nyquist = 0.5 * sampling_rate
        if filter_type == "low":
            b, a = butter(order, cutoff / nyquist, btype="low")
        elif filter_type == "high":
            b, a = butter(order, cutoff / nyquist, btype="high")
        elif filter_type == "band":
            b, a = butter(order, [cutoff[0] / nyquist, cutoff[1] / nyquist], btype="band")
        else:
            raise ValueError("Geçersiz filtre türü! 'low', 'high', veya 'band' olmalı.")

        filtered_signal = filtfilt(b, a, self.data[channel_name])
        return filtered_signal

    def filter_all_channels(self, filter_type, cutoff, order=4, sampling_rate=1000):
        """
        Tüm kanallara filtre uygular ve filtrelenmiş veriyi saklar.
        :param filter_type: Filtre türü ('low', 'high', veya 'band')
        :param cutoff: Kesim frekansı (low/high için tek değer, band için [low, high])
        :param order: Filtre düzeni
        :param sampling_rate: Örnekleme frekansı
        """
        self.filtered_data = self.data.copy()
        for channel in self.channels:
            self.filtered_data[channel] = self.apply_filter(channel, filter_type, cutoff, order, sampling_rate)
        print("Tüm kanallar filtrelendi.")

    def save_filtered_data(self, output_csv_path):
        """
        Filtrelenmiş veriyi bir CSV dosyasına kaydeder.
        :param output_csv_path: Kaydedilecek dosya yolu
        """
        if self.filtered_data is not None:
            self.filtered_data.to_csv(output_csv_path, index=False)
            print(f"Filtrelenmiş veri kaydedildi: {output_csv_path}")
        else:
            print("Filtrelenmiş veri bulunamadı. Lütfen önce 'filter_all_channels' fonksiyonunu çalıştırın.")

    def visualize_filtered_data(self, output_plot_path, start=0, end=1000):
        """
        Orijinal ve filtrelenmiş sinyalleri karşılaştıran bir görselleştirme oluşturur.
        :param output_plot_path: Görselleştirmenin kaydedileceği PNG dosya yolu
        :param start: Başlangıç indeksi
        :param end: Bitiş indeksi
        """
        if self.filtered_data is None:
            print("Filtrelenmiş veri bulunamadı. Lütfen önce 'filter_all_channels' fonksiyonunu çalıştırın.")
            return

        fig, axes = plt.subplots(len(self.channels), 1, figsize=(12, len(self.channels) * 3))
        if len(self.channels) == 1:
            axes = [axes]

        for i, channel in enumerate(self.channels):
            axes[i].plot(self.data[channel][start:end], label="Orijinal", alpha=0.7)
            axes[i].plot(self.filtered_data[channel][start:end], label="Filtrelenmiş", alpha=0.7)
            axes[i].set_title(f"{channel} - Filtrelenmiş Karşılaştırması")
            axes[i].set_xlabel("Zaman")
            axes[i].set_ylabel("Amplitüd")
            axes[i].legend()
            axes[i].grid()

        plt.tight_layout()
        plt.savefig(output_plot_path)
        print(f"Görselleştirme kaydedildi: {output_plot_path}")
        plt.close()


if __name__ == "__main__":
    # Dengelenmiş veri setini yükle
    file_path = "./dataset/Balanced_EMG_data.csv"
    data = pd.read_csv(file_path)

    # EMG kanalları
    channels = [f"channel{i}" for i in range(1, 9)]  # channel1, channel2, ..., channel8

    # EMGSignalProcessor sınıfını başlat
    processor = EMGSignalProcessor(data, channels)

    # Filtreleme işlemi
    processor.filter_all_channels(
        filter_type="low",
        cutoff=50,
        order=4,
        sampling_rate=1000
    )

    # Filtrelenmiş veriyi kaydet
    processor.save_filtered_data("./dataset/filtered_balanced_emg_data.csv")

    # Orijinal ve filtrelenmiş veriyi görselleştir
    processor.visualize_filtered_data("./dataset/filtered_emg_visualization.png", start=0, end=1000)
