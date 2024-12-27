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
    import os
    from data_processor import DataProcessor
    from data_filter import DataFilter

    # 1. Ham Veriyi Görselleştirme
    raw_processor = DataProcessor(["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"])
    raw_processor.set_data_path("dataset/emg_data.csv")
    raw_processor.set_save_path("results/raw_data_visualizations")
    raw_processor.load_data()
    raw_processor.visualize_each_class()

    # 2. Filtreleme İşlemi
    print("\n--- Filtreleme işlemi başlıyor ---")
    filter_processor = DataFilter(
        input_path="dataset/emg_data.csv",
        output_path="filtered_dataset/emg_filtered_data.csv",
        lowcut=5,
        highcut=90,
        fs=200,
        order=4,
    )
    filter_processor.load_data()  # Ham veriyi yükle
    filtered_dataset = filter_processor.filter_data()  # Filtre uygula
    filter_processor.save_data()  # Filtrelenmiş veriyi kaydet

    # 3. Filtrelenmiş Veriyi Görselleştirme
    filtered_processor = DataProcessor(["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"])
    filtered_processor.set_data_path("filtered_dataset/emg_filtered_data.csv")
    filtered_processor.set_save_path("results/filtered_data_visualizations")
    filtered_processor.load_data()
    filtered_processor.visualize_each_class()

    print("\n--- Tüm işlemler başarıyla tamamlandı ---")
