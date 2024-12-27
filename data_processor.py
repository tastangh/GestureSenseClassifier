import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

class DataProcessor:
    def __init__(self, data_path, class_names):
        """
        :param data_path: Karıştırılmış veri dosyasının yolu
        :param class_names: Sınıf isimlerinin listesi
        """
        self.data_path = data_path
        self.class_names = class_names
        self.dataset = None

    def load_data(self):
        """Tek bir veri dosyasını yükler."""
        if os.path.exists(self.data_path):
            print(f"{self.data_path} yükleniyor...")
            self.dataset = pd.read_csv(self.data_path)
            print("Veri başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"{self.data_path} dosyası bulunamadı!")
        return self.dataset

    def plot_40ms_data(self, data_row, class_name):
        """
        Her sensörün 40 ms boyunca alınan 8 ölçümünü aynı grafikte gösterir.
        :param data_row: Görselleştirilecek satır
        :param class_name: Sınıf adı (ör. "Taş(0)")
        """
        save_path = "results/dataset"
        os.makedirs(save_path, exist_ok=True)

        x_time = np.linspace(5, 40, 8)  # 200 hz'lık 8 ölçüm noktası

        plt.figure(figsize=(12, 6))
        for i in range(8):  # 8 sensör
            sensor_data = data_row[i::8].values  # Sensöre ait ölçümler
            plt.plot(x_time, sensor_data, label=f"Sensor {i + 1}")

        plt.title(f"{class_name} Hareketi - 40 ms Ardışık Ölçümleri")
        plt.xlabel("Zaman (ms)")
        plt.ylabel("Örnek Değeri")
        plt.legend(loc="upper right")
        plt.grid(True)

        save_file = os.path.join(save_path, f"{class_name.lower().replace('(', '').replace(')', '').replace(' ', '_')}_40ms.png")
        plt.savefig(save_file)
        plt.show()
        print(f"{class_name} sınıfı için figür {save_file} konumuna kaydedildi.")

    def plot_heatmap(self, data_row, class_name):
        """
        Sensör ölçümlerini 8x8 ısı haritası olarak görselleştirir.
        """
        save_path = "results/dataset"
        os.makedirs(save_path, exist_ok=True)

        heatmap_data = data_row.values.reshape(8, 8)
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap_data, cmap="viridis", aspect="auto")
        plt.colorbar(label="Örnek Değeri")
        plt.title(f"{class_name} Hareketi - Isı Haritası")
        plt.xlabel("Zaman Noktası")
        plt.ylabel("Sensör Numarası")

        save_file = os.path.join(save_path, f"{class_name.lower().replace('(', '').replace(')', '').replace(' ', '_')}_heatmap.png")
        plt.savefig(save_file)
        plt.show()
        print(f"{class_name} sınıfı için ısı haritası {save_file} konumuna kaydedildi.")

    def visualize_each_class(self):
        """Her Gesture_Class için bir örnek seçer ve görselleştirir."""
        if self.dataset is None:
            raise ValueError("Veri seti yüklenmemiş. Önce `load_data` çağrılmalı.")
        
        gesture_classes = self.dataset["Gesture_Class"].unique()
        for gesture_class in gesture_classes:
            sample_row = self.dataset[self.dataset["Gesture_Class"] == gesture_class].iloc[0, :-1]
            class_name = self.class_names[int(gesture_class)]
            print(f"{class_name} sınıfı görselleştiriliyor...")
            self.plot_40ms_data(sample_row, class_name)
            self.plot_heatmap(sample_row, class_name)

if __name__ == "__main__":
    # Veri dosyasının yolu ve sınıf isimleri
    data_path = "dataset/emg_data.csv"
    class_names = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # DataProcessor nesnesi oluştur
    processor = DataProcessor(data_path, class_names)

    # Veriyi yükle
    dataset = processor.load_data()

    # Her Gesture_Class için bir örneği görselleştir
    processor.visualize_each_class()

    print("\n--- Tüm görselleştirme işlemleri tamamlandı ---")
