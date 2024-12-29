import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os


class EMGSignalProcessor:
    def __init__(self, data, channels, sampling_rate=1000):
        """
        EMG sinyali filtreleme işlemleri için sınıf.
        :param data: Veri seti (pandas DataFrame)
        :param channels: EMG kanalları
        :param sampling_rate: Örnekleme frekansı
        """
        self.data = data
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.filtered_data = data.copy()

    @staticmethod
    def apply_filter(signal, filter_type, cutoff, order=4, sampling_rate=1000):
        """
        Sinyale filtre uygular.
        :param signal: Tek kanal sinyali (numpy array)
        :param filter_type: 'low', 'high', veya 'band'
        :param cutoff: Kesim frekansı (low/high için tek değer, band için [low, high])
        :param order: Filtre düzeni
        :param sampling_rate: Örnekleme frekansı
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
        else:
            raise ValueError("Invalid filter type! Use 'low', 'high', or 'band'.")
        return filtfilt(b, a, signal)

    def filter_all_channels(self, filter_type="band", cutoff=(20, 450), order=4):
        """
        Tüm kanallara filtre uygular ve filtrelenmiş veriyi saklar.
        :param filter_type: Filtre türü ('low', 'high', veya 'band')
        :param cutoff: Kesim frekansı
        :param order: Filtre düzeni
        """
        for channel in self.channels:
            self.filtered_data[channel] = self.apply_filter(
                self.data[channel], filter_type, cutoff, order, self.sampling_rate
            )
        print(f"Tüm kanallar için {filter_type} filtre uygulandı.")

    def plot_frequency_spectrum(self, signals, filtered_signals, titles, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
        axes = axes.ravel()

        for i, (signal, filtered_signal, title) in enumerate(zip(signals, filtered_signals, titles)):
            N = len(signal)
            T = 1 / self.sampling_rate
            yf_original = np.abs(np.fft.fft(signal))[:N // 2]
            yf_filtered = np.abs(np.fft.fft(filtered_signal))[:N // 2]
            xf = np.fft.fftfreq(N, T)[:N // 2]

            axes[i].plot(xf, yf_original, label="Original", alpha=0.7)
            axes[i].plot(xf, yf_filtered, label="Filtered", alpha=0.7)
            axes[i].set_title(title, fontsize=10, pad=10)
            axes[i].set_xlabel("Frequency (Hz)", fontsize=8, labelpad=5)
            axes[i].set_ylabel("Amplitude", fontsize=8, labelpad=5)
            axes[i].legend(fontsize=8)
            axes[i].grid()

        for ax in axes[len(signals):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Frequency spectra saved to: {output_path}")
        plt.close()

    def plot_signals(self, signals, filtered_signals, titles, output_path, start=0, end=1000):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
        axes = axes.ravel()

        for i, (signal, filtered_signal, title) in enumerate(zip(signals, filtered_signals, titles)):
            axes[i].plot(signal[start:end], label="Original", alpha=0.7)
            axes[i].plot(filtered_signal[start:end], label="Filtered", alpha=0.7)
            axes[i].set_title(title, fontsize=10, pad=10)
            axes[i].set_xlabel("Time", fontsize=8, labelpad=5)
            axes[i].set_ylabel("Amplitude", fontsize=8, labelpad=5)
            axes[i].legend(fontsize=8)
            axes[i].grid()

        for ax in axes[len(signals):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Signals saved to: {output_path}")
        plt.close()

    def get_filtered_data(self):
        return self.filtered_data


if __name__ == "__main__":
    file_path = "./dataset/EMG-data.csv"
    data = pd.read_csv(file_path)
    print("Dataset successfully loaded.")

    channels = [f"channel{i}" for i in range(1, 9)]
    sampling_rate = 1000

    processor = EMGSignalProcessor(data, channels, sampling_rate=sampling_rate)

    print("\nApplying band-pass filter to all channels...")
    processor.filter_all_channels(filter_type="band", cutoff=(20, 450), order=4)

    print("\nPlotting frequency spectra for all channels...")
    processor.plot_frequency_spectrum(
        signals=[data[channel] for channel in channels],
        filtered_signals=[processor.filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Frequency Spectrum" for channel in channels],
        output_path="./output/frequency_spectra.png"
    )

    print("\nPlotting time-domain signals for all channels...")
    processor.plot_signals(
        signals=[data[channel] for channel in channels],
        filtered_signals=[processor.filtered_data[channel] for channel in channels],
        titles=[f"{channel} - Time-Domain Signals" for channel in channels],
        output_path="./output/time_domain_signals.png",
        start=0,
        end=1000
    )

    output_path = "./dataset/filtered_emg_data.csv"
    processor.get_filtered_data().to_csv(output_path, index=False)
    print(f"Filtered dataset saved to: {output_path}")
