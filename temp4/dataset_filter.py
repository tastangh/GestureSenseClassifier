# dataset_filter.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import os


class DatasetFilter:
    """
    Dataset üzerinde filtreleme ve görselleştirme işlemleri için sınıf.
    """

    def __init__(self, data, channels, sampling_rate=1000):
        """
        DatasetFilter sınıfının başlatıcı metodu.
        :param data: Veri seti (pandas DataFrame)
        :param channels: Filtreleme yapılacak kanallar
        :param sampling_rate: Örnekleme frekansı
        """
        self.data = data
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.filtered_data = data.copy()

    @staticmethod
    def apply_filter(signal, filter_type, cutoff, order=4, sampling_rate=1000, notch_freq=None):
        """
        Sinyale filtre uygular.
        :param signal: Tek kanal sinyali (numpy array)
        :param filter_type: 'low', 'high', 'band', veya 'notch'
        :param cutoff: Kesim frekansı (low/high için tek değer, band için [low, high])
        :param order: Filtre düzeni
        :param sampling_rate: Örnekleme frekansı
        :param notch_freq: Notch filtre için kesim frekansı (isteğe bağlı)
        :return: Filtrelenmiş sinyal
        """
        nyquist = 0.5 * sampling_rate
        if filter_type == "low":
            normalized_cutoff = cutoff / nyquist
            b, a = butter(order, normalized_cutoff, btype="low")
        elif filter_type == "high":
            normalized_cutoff = cutoff / nyquist
            b, a = butter(order, normalized_cutoff, btype="high")
        elif filter_type == "band":
            normalized_cutoff = [freq / nyquist for freq in cutoff]
            b, a = butter(order, normalized_cutoff, btype="band")
        elif filter_type == "notch" and notch_freq is not None:
            Q = 30  # Notch filtre kalitesi
            normalized_freq = notch_freq / nyquist
            b, a = iirnotch(normalized_freq, Q, sampling_rate)
        else:
            raise ValueError("Hatalı filtre türü! 'low', 'high', 'band' veya 'notch' kullanın.")
        return filtfilt(b, a, signal)
    

    def filter_all_channels(self, filter_type="band", cutoff=(20, 450), order=4, apply_notch=False, notch_freq=50):
        """
        Tüm kanallara filtre uygular ve filtrelenmiş veriyi saklar.
        :param filter_type: Filtre türü ('low', 'high', veya 'band')
        :param cutoff: Kesim frekansı
        :param order: Filtre düzeni
        :param apply_notch: Notch filtresi uygulanıp uygulanmayacağı (bool)
        :param notch_freq: Notch filtre frekansı (Hz)
        """
        for channel in self.channels:
            self.filtered_data[channel] = self.apply_filter(
                self.data[channel], filter_type, cutoff, order, self.sampling_rate
            )
        
        if apply_notch:
            for channel in self.channels:
                 self.filtered_data[channel] = self.apply_filter(
                self.filtered_data[channel], "notch", None, order=4, sampling_rate=self.sampling_rate, notch_freq=notch_freq
            )
        print(f"Tüm kanallar için {filter_type} filtre uygulandı.")
        if apply_notch:
          print(f"Tüm kanallar için {notch_freq}Hz notch filtresi uygulandı.")


    def plot_frequency_spectrum(self, signals, filtered_signals, titles, output_path):
        """
        Frekans spektrumlarını çizer ve kaydeder.
        :param signals: Orijinal sinyallerin listesi
        :param filtered_signals: Filtrelenmiş sinyallerin listesi
        :param titles: Her kanal için başlık listesi
        :param output_path: Grafiğin kaydedileceği yol
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
        axes = axes.ravel()

        for i, (signal, filtered_signal, title) in enumerate(zip(signals, filtered_signals, titles)):
            N = len(signal)
            T = 1 / self.sampling_rate
            yf_original = np.abs(np.fft.fft(signal))[:N // 2]
            yf_filtered = np.abs(np.fft.fft(filtered_signal))[:N // 2]
            xf = np.fft.fftfreq(N, T)[:N // 2]

            axes[i].plot(xf, yf_original, label="Orijinal", alpha=0.7)
            axes[i].plot(xf, yf_filtered, label="Filtrelenmiş", alpha=0.7)
            axes[i].set_title(title, fontsize=10, pad=10)
            axes[i].set_xlabel("Frekans (Hz)", fontsize=8, labelpad=5)
            axes[i].set_ylabel("Genlik", fontsize=8, labelpad=5)
            axes[i].legend(fontsize=8)
            axes[i].grid()
            axes[i].set_xscale('log') # Logaritmik eksen

        for ax in axes[len(signals):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Frekans spektrum grafikleri kaydedildi: {output_path}")
        plt.close()

    def plot_signals(self, signals, filtered_signals, titles, output_path, start=0, end=1000):
        """
        Zaman domenindeki sinyalleri çizer ve kaydeder.
        :param signals: Orijinal sinyallerin listesi
        :param filtered_signals: Filtrelenmiş sinyallerin listesi
        :param titles: Her kanal için başlık listesi
        :param output_path: Grafiğin kaydedileceği yol
        :param start: Çizimin başlangıç indeksi
        :param end: Çizimin bitiş indeksi
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
        axes = axes.ravel()

        for i, (signal, filtered_signal, title) in enumerate(zip(signals, filtered_signals, titles)):
            axes[i].plot(signal[start:end], label="Orijinal", alpha=0.7)
            axes[i].plot(filtered_signal[start:end], label="Filtrelenmiş", alpha=0.7)
            axes[i].set_title(title, fontsize=10, pad=10)
            axes[i].set_xlabel("Zaman", fontsize=8, labelpad=5)
            axes[i].set_ylabel("Genlik", fontsize=8, labelpad=5)
            axes[i].legend(fontsize=8)
            axes[i].grid()

        for ax in axes[len(signals):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Zaman domeni sinyal grafikleri kaydedildi: {output_path}")
        plt.close()

    def get_filtered_data(self):
        """
        Filtrelenmiş veriyi döndürür.
        :return: pandas DataFrame (Filtrelenmiş veri)
        """
        return self.filtered_data

if __name__ == "__main__":
    file_path = "./dataset/EMG-data.csv"
    data = pd.read_csv(file_path)
    print("Veri seti başarıyla yüklendi.")

    channels = [f"channel{i}" for i in range(1, 9)]
    sampling_rate = 1000
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' klasörü oluşturuldu.")

    processor = DatasetFilter(data, channels, sampling_rate=sampling_rate)

    print("\nTüm kanallar için band geçiren filtre uygulanıyor...")
    processor.filter_all_channels(filter_type="band", cutoff=(20, 450), order=4, apply_notch=True, notch_freq=50)

    print("\nTüm kanallar için frekans spektrumları çiziliyor...")
    processor.plot_frequency_spectrum(
        signals=[data[channel] for channel in channels],
        filtered_signals=[processor.filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Frekans Spektrumu" for channel in channels],
        output_path="./output/frequency_spectra.png"
    )

    print("\nTüm kanallar için zaman domeni sinyalleri çiziliyor...")
    processor.plot_signals(
        signals=[data[channel] for channel in channels],
        filtered_signals=[processor.filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Zaman Domeni Sinyalleri" for channel in channels],
        output_path="./output/time_domain_signals.png",
        start=0,
        end=1000
    )
    
    output_path = os.path.join(output_dir,"filtered_emg_data.csv")
    processor.get_filtered_data().to_csv(output_path, index=False)
    print(f"Filtrelenmiş veri seti kaydedildi: {output_path}")