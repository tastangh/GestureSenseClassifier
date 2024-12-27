import os
import pandas as pd
from scipy.signal import butter, filtfilt
from data_processor import DataProcessor

class DataFilter:
    def __init__(self, input_path, output_path, lowcut=5, highcut=90, fs=200, order=4):
        """
        :param input_path: Girdi veri dosyasının yolu.
        :param output_path: Filtrelenmiş verinin kaydedileceği yol.
        :param lowcut: Bant geçiren filtrenin alt kesme frekansı (Hz).
        :param highcut: Bant geçiren filtrenin üst kesme frekansı (Hz).
        :param fs: Örnekleme frekansı (Hz).
        :param order: Filtrenin derecesi.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.dataset = None

    def load_data(self):
        """Veriyi yükler."""
        if os.path.exists(self.input_path):
            print(f"{self.input_path} yükleniyor...")
            self.dataset = pd.read_csv(self.input_path)
            print("Veri başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"{self.input_path} dosyası bulunamadı!")
        return self.dataset

    def bandpass_filter(self, data):
        """Bant geçiren filtre uygular."""
        nyquist = 0.5 * self.fs  # Nyquist frekansı
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def filter_data(self):
        """Veriye filtre uygular."""
        if self.dataset is None:
            raise ValueError("Veri yüklenmedi. Önce `load_data` çağrılmalı.")
        
        # Sensör verilerini filtreleme
        sensor_data = self.dataset.iloc[:, :-1]  # Son sütun hariç (Gesture_Class)
        gesture_class = self.dataset["Gesture_Class"]  # Sınıf etiketleri
        print("Filtreleme işlemi başlatılıyor...")
        filtered_sensor_data = self.bandpass_filter(sensor_data.values)
        
        # Filtrelenmiş veriyi birleştirme
        filtered_dataset = pd.DataFrame(filtered_sensor_data, columns=sensor_data.columns)
        filtered_dataset["Gesture_Class"] = gesture_class
        self.dataset = filtered_dataset
        print("Filtreleme işlemi tamamlandı.")
        return self.dataset

    def save_data(self):
        """Filtrelenmiş veriyi kaydeder."""
        if self.dataset is None:
            raise ValueError("Filtrelenmiş veri mevcut değil.")
        
        # Kaydetme işlemi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.dataset.to_csv(self.output_path, index=False)
        print(f"Filtrelenmiş veri {self.output_path} dosyasına kaydedildi.")

# Kullanım Örneği
if __name__ == "__main__":
    # Veri dosyalarının yolları ve sınıf isimleri
    raw_data_path = "dataset/emg_data.csv"
    filtered_data_path = "filtered_dataset/emg_filtered.csv"
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # 1. DataFilter: Filtreleme işlemi
    print("\n--- Filtreleme İşlemi ---")
    data_filter = DataFilter(raw_data_path, filtered_data_path)
    data_filter.load_data()
    data_filter.filter_data()
    data_filter.save_data()

    # 2. DataProcessor: Görselleştirme işlemi
    print("\n--- Görselleştirme İşlemi (Filtrelenmiş Veri) ---")
    processor = DataProcessor(filtered_data_path, class_names)
    processor.load_data()  # Varsayılan olarak filtrelenmiş veriyi yükler
    processor.visualize_each_class()

    print("\n--- Tüm işlemler başarıyla tamamlandı ---")
