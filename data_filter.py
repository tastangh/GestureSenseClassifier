import pandas as pd
from scipy.signal import butter, lfilter, iirnotch
import matplotlib.pyplot as plt
import os

class DataFilter:
    def __init__(self, data, sampling_rate=1000):
        """
        DataFilter sınıfı, veri setini filtrelemek için kullanılır.
        :param data: Veri seti (pandas DataFrame)
        :param sampling_rate: Örnekleme frekansı (Hz)
        """
        self.data = data
        self.sampling_rate = sampling_rate

    def bandpass_filter(self, column, lowcut, highcut, order=4):
        """
        Bant geçiş filtresi uygular.
        :param column: Filtrelenecek sütun adı
        :param lowcut: Alt kesme frekansı (Hz)
        :param highcut: Üst kesme frekansı (Hz)
        :param order: Filtre sırası
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        self.data[column] = lfilter(b, a, self.data[column])
        print(f"Bant geçiş filtresi uygulandı: {column} ({lowcut}-{highcut} Hz)")

    def notch_filter(self, column, freq, quality=30):
        """
        Notch filtresi uygular (örneğin, 50 Hz hat gürültüsü).
        :param column: Filtrelenecek sütun adı
        :param freq: Notch frekansı (Hz)
        :param quality: Quality faktörü
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = freq / nyquist
        b, a = iirnotch(w0, quality)
        self.data[column] = lfilter(b, a, self.data[column])
        print(f"Notch filtresi uygulandı: {column} ({freq} Hz)")

    def apply_filters(self, columns, filter_type="both", bandpass_params=None, notch_params=None):
        """
        Belirtilen sütunlara filtreler uygular.
        :param columns: Filtrelenecek sütunların listesi
        :param filter_type: Filtre türü ("band", "notch", "both", "none")
        :param bandpass_params: Bant geçiş filtresi parametreleri (lowcut, highcut, order)
        :param notch_params: Notch filtresi parametreleri (freq, quality)
        """
        for column in columns:
            if filter_type in ["band", "both"] and bandpass_params:
                self.bandpass_filter(column, **bandpass_params)
            if filter_type in ["notch", "both"] and notch_params:
                self.notch_filter(column, **notch_params)
            if filter_type == "none":
                print(f"Hiçbir filtre uygulanmadı: {column}")

    def plot_signals_comparison(self, original_data, columns, output_dir, start=0, end=1000, filter_params=None):
        """
        Orijinal ve filtrelenmiş sinyalleri karşılaştırmalı olarak 2x4 düzeninde kaydeder.
        :param original_data: Filtre öncesi veri seti (pandas DataFrame)
        :param columns: Çizilecek sütunların listesi
        :param output_dir: Grafiklerin kaydedileceği klasör
        :param start: Başlangıç indeksi
        :param end: Bitiş indeksi
        :param filter_params: Filtre parametreleri (bandpass_params, notch_params)
        """
        os.makedirs(output_dir, exist_ok=True)  # Klasörü oluştur

        # Parametreleri dosya adı için birleştir
        params_stamp = "_".join([f"{key}_{value}" for key, value in (filter_params or {}).items()])
        plot_filename = f"comparison_signals_{params_stamp}.png"

        num_rows, num_cols = 4, 2  # 2 sütun, 4 satır
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # 2D matristen 1D listeye dönüştür

        for i, column in enumerate(columns):
            ax = axes[i]
            ax.plot(original_data[column][start:end], label='Orijinal', alpha=0.7)
            ax.plot(self.data[column][start:end], label='Filtrelenmiş', alpha=0.7, linestyle='--')
            ax.set_title(f"{column} - Orijinal vs. Filtrelenmiş")
            ax.set_xlabel("Örnek")
            ax.set_ylabel("Genlik")
            ax.legend()
            ax.grid(True)

        # Kalan boş grafik alanlarını gizle
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        output_path = os.path.join(output_dir, plot_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()  # Grafiği kapat
        print(f"Karşılaştırmalı sinyaller kaydedildi: {output_path}")


if __name__ == "__main__":
    # Veri setinin yolu
    dataset_path = "dataset/balanced_EMG_data.csv"
    sampling_rate = 1000  # Örnekleme frekansı
    channels = [f"channel{i}" for i in range(1, 9)]  # Kanal isimleri

    # Veri yükleme
    print("Veri yükleniyor...")
    data = pd.read_csv(dataset_path)
    print(f"Veri başarıyla yüklendi. Toplam satır: {len(data)}")

    # Orijinal veri yedeği
    original_data = data.copy()

    # Filtreleme sınıfını başlat
    data_filter = DataFilter(data, sampling_rate=sampling_rate)

    # Filtreleme parametreleri
    bandpass_params = {"lowcut": 20, "highcut": 450, "order": 4}
    notch_params = {"freq": 50, "quality": 30}
    filter_params = {**bandpass_params, **notch_params}  # Parametreleri birleştir

    # Filtre türünü seç
    filter_type = "both"  # "band", "notch", "both", "none"

    # Filtreleri uygula
    print("\nFiltreler uygulanıyor...")
    data_filter.apply_filters(columns=channels, filter_type=filter_type, bandpass_params=bandpass_params, notch_params=notch_params)

    # Orijinal ve filtrelenmiş sinyalleri karşılaştır
    output_dir = "plots"
    print("\nOrijinal ve Filtrelenmiş sinyaller karşılaştırılıyor...")
    data_filter.plot_signals_comparison(original_data, columns=channels, output_dir=output_dir, start=0, end=1000, filter_params=filter_params)

    # Filtrelenmiş veriyi kaydet
    output_path = "dataset/filtered_balanced_EMG_data.csv"
    data_filter.data.to_csv(output_path, index=False)
    print(f"Filtrelenmiş veri kaydedildi: {output_path}")
