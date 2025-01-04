import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from pandas.plotting import scatter_matrix
import seaborn as sns

class Visualizer:
    """
    Veri işleme ve modelleme süreçlerini görselleştirmek için sınıf.
    """
    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_plot(self, filename):
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        print(f"Görselleştirme kaydedildi: {save_path}")
        plt.close()

    def plot_class_distribution(self, data, class_column="class", title="Class Distribution", filename="class_distribution.png", show=False):
        """
        Sınıf dağılımını çizer ve her çubuğun üstüne örnek sayısını ekler.
        """
        if isinstance(data, pd.DataFrame):
            class_counts = data[class_column].value_counts()
        elif isinstance(data, (list, pd.Series)):
            class_counts = pd.Series(data).value_counts()
        else:
            raise ValueError("Data must be a pandas DataFrame, list, or pandas Series.")

        plt.figure(figsize=(8, 6))
        ax = class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Classes")
        plt.ylabel("Counts")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Çubukların üstüne örnek sayılarını yaz
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), textcoords='offset points')

        self._save_plot(filename)
        if show:
            plt.show()
            
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

    def plot_filtered_signals(self, original_data, filtered_data, channels, sampling_rate, output_prefix, start=0, end=1000):
        """
        Zaman ve frekans domeni sinyallerini tek bir fonksiyon ile çizer.
        :param original_data: Orijinal veri seti (pandas DataFrame)
        :param filtered_data: Filtrelenmiş veri seti (pandas DataFrame)
        :param channels: İşlenecek kanalların listesi
        :param sampling_rate: Örnekleme frekansı
        :param output_prefix: Çıktı dosyaları için önek
        :param start: Zaman domeni için başlangıç indeksi
        :param end: Zaman domeni için bitiş indeksi
        """
        # Zaman Domeni Görselleştirme
        print("\nZaman domeni sinyalleri çiziliyor...")
        self.plot_signals(
            signals=[original_data[channel].values for channel in channels],
            filtered_signals=[filtered_data[channel].values for channel in channels],
            titles=[f"{channel} - Zaman Domeni" for channel in channels],
            output_path=os.path.join(self.output_dir, f"{output_prefix}_time_domain.png"),
            start=start,
            end=end
        )

        # Frekans Spektrumu Görselleştirme
        print("\nFrekans spektrumu çiziliyor...")
        fig, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
        axes = axes.ravel()

        for i, channel in enumerate(channels):
            # Orijinal ve filtrelenmiş sinyallerin Fourier Transform'u
            signal = original_data[channel].values
            filtered_signal = filtered_data[channel].values
            N = len(signal)
            T = 1 / sampling_rate
            freqs = np.fft.fftfreq(N, T)[:N // 2]
            fft_original = np.abs(np.fft.fft(signal))[:N // 2]
            fft_filtered = np.abs(np.fft.fft(filtered_signal))[:N // 2]

            # Frekans Spektrumu Çizimi
            axes[i].plot(freqs, fft_original, label="Orijinal", alpha=0.7)
            axes[i].plot(freqs, fft_filtered, label="Filtrelenmiş", alpha=0.7)
            axes[i].set_title(f"{channel} - Frekans Spektrumu", fontsize=10, pad=10)
            axes[i].set_xlabel("Frekans (Hz)", fontsize=8, labelpad=5)
            axes[i].set_ylabel("Genlik", fontsize=8, labelpad=5)
            axes[i].legend(fontsize=8)
            axes[i].grid()

        # Boş grafikleri kaldırma
        for ax in axes[len(channels):]:
            ax.axis('off')

        # Grafiği kaydetme
        output_path = os.path.join(self.output_dir, f"{output_prefix}_frequency_spectrum.png")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Frekans spektrumu grafikleri kaydedildi: {output_path}")
        plt.close()
