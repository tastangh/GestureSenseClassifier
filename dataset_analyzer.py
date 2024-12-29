import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
import numpy as np

class DatasetAnalyzer:
    def _init_(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Veri setini yükler."""
        self.data = pd.read_csv(self.file_path)
        print(f"Veri seti başarıyla yüklendi: {self.file_path}")

    def basic_info(self):
        """Veri seti hakkında temel bilgileri görüntüler."""
        if self.data is not None:
            print("\nVeri Seti Bilgileri:")
            print(self.data.info())
            print("\nVeri Setinin İlk 5 Satırı:")
            print(self.data.head())
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")

    def check_missing_values(self):
        """Veri setindeki eksik değerleri kontrol eder."""
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print("\nHer Sütundaki Eksik Değer Sayısı:")
            print(missing_values)
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")
            
    def check_noise(self, signal_column):
        """Veri setindeki sinyaldeki noise'u kontrol eder."""
        if self.data is not None:
            if signal_column in self.data.columns:
                signal = self.data[signal_column].values

                # 1. Temel İstatistiksel Özellikler
                mean = np.mean(signal)
                std_dev = np.std(signal)
                print(f"Sütun: {signal_column}")
                print(f"Ortalama: {mean}")
                print(f"Standart Sapma: {std_dev}")

                # 2. Fourier Dönüşümü
                N = len(signal)
                T = 1.0 / 1000.0  # Örnekleme aralığı (örneğin 1000 Hz)
                yf = fft(signal)
                xf = fftfreq(N, T)[:N // 2]

                plt.figure(figsize=(10, 6))
                plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
                plt.title(f"Frekans Spektrumu: {signal_column}", fontsize=16)
                plt.xlabel("Frekans (Hz)", fontsize=12)
                plt.ylabel("Amplitüd", fontsize=12)
                plt.grid()
                plt.show()

                # 3. Sinyal-Gürültü Oranı (SNR)
                signal_power = np.mean(signal ** 2)
                noise_power = np.var(signal - np.mean(signal))
                snr = 10 * np.log10(signal_power / noise_power)
                print(f"Sinyal-Gürültü Oranı (SNR): {snr:.2f} dB")

                # Noise kontrolü
                noise_threshold = 10  # SNR eşik değeri (dB)
                has_noise = snr < noise_threshold

                # 4. Dalga Dönüşümü (Wavelet Transform)
                coeffs = pywt.wavedec(signal, 'db4', level=5)
                plt.figure(figsize=(10, 6))
                for i, coeff in enumerate(coeffs[1:], start=1):
                    plt.plot(coeff, label=f"Detay Katsayısı {i}")
                plt.title(f"Wavelet Detay Katsayıları: {signal_column}", fontsize=16)
                plt.xlabel("Zaman", fontsize=12)
                plt.ylabel("Amplitüd", fontsize=12)
                plt.legend()
                plt.grid()
                plt.show()

                return has_noise

            else:
                print(f"'{signal_column}' adlı sütun veri setinde bulunamadı.")
                return False
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")
            return False

    def visualize_class_distribution(self):
        """Sınıf dağılımını grafik ve sayılarla görselleştirir."""
        if self.data is not None:
            if "class" in self.data.columns:
                class_counts = self.data["class"].value_counts().sort_index()
                
                # Bar grafiği çiz
                ax = class_counts.plot(kind="bar", figsize=(10, 6), color="skyblue", edgecolor="black")
                plt.title("Sınıf Dağılımı", fontsize=16)
                plt.xlabel("Sınıf", fontsize=12)
                plt.ylabel("Frekans", fontsize=12)
                
                # Çubukların üzerine sayı etiketleri ekle
                for i, count in enumerate(class_counts):
                    plt.text(i, count + 500, str(count), ha='center', fontsize=10)
                
                plt.xticks(rotation=0)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.tight_layout()
                plt.show()
                
                # Dağılımı tablo olarak yazdır
                print("\nSınıf Dağılımı (Sayısal):")
                print(class_counts)
            else:
                print("Veri setinde 'class' sütunu bulunamadı.")
        else:
            print("Veri henüz yüklenmedi. Lütfen önce veri yükleyin.")


if __name__ == "__main__":
    # Veri seti dosyasının yolu
    file_path = "./dataset/EMG-data.csv"  # Dosya yolunu ayarlayın

    analyzer = DatasetAnalyzer(file_path)
    # Veri analizi gerçekleştir
    analyzer.load_data()
    analyzer.basic_info()
    analyzer.check_missing_values()
    analyzer.visualize_class_distribution()

   # Noise kontrolü
    signal_columns = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"]
    for column in signal_columns:
        print(f"\n{column} sütunu için noise analizi:")
        analyzer.check_noise(column)