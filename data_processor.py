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
        Her sensörün 40 ms boyunca alınan 8 ölçümünü sabit bir çizgi şeklinde gösterir.
        :param data_row: Görselleştirilecek satır
        :param class_name: Sınıf adı (ör. "Taş(0)")
        """
        save_path = "results/dataset"
        os.makedirs(save_path, exist_ok=True)

        # 8 ölçüm, her ölçüm 5 ms arayla
        x_time = np.linspace(0, 40, 9)  # 9 noktadan oluşuyor, her ölçüm 5 ms arayla

        # 8 sensör için alt grafikler (subplot) oluştur
        plt.figure(figsize=(16, 12))  # Tüm sensörlerin grafiklerini yerleştirecek kadar geniş bir figür

        for i in range(8):  # 8 sensör
            sensor_data = data_row[i::8].values  # Sensöre ait ölçümler
            
            # Her sensör için bir subplot oluştur
            plt.subplot(4, 2, i + 1)  # 4 satır, 2 sütunluk alt grafikler düzeni

            # 8 ölçüm, her biri 5 ms süresince sabit
            for j in range(1, len(x_time)):
                # Her ölçüm için sabit kalacak şekilde çizim (5 ms boyunca sabit)
                plt.plot([x_time[j-1], x_time[j]], [sensor_data[j-1], sensor_data[j-1]], label=f"Sensor {i + 1}" if j == 1 else "")

            # Her sensör için başlık, etiketler ve grid ekle
            plt.title(f"Sensor {i + 1} - {class_name} Hareketi")
            plt.xlabel("Zaman (ms)")
            plt.ylabel("Örnek Değeri")
            plt.grid(True)
            plt.tight_layout()  # Alt grafiklerin düzenini sıkıştırarak daha düzgün bir yerleşim sağlar

        # Grafikleri kaydet
        save_file = os.path.join(save_path, f"{class_name.lower().replace('(', '').replace(')', '').replace(' ', '_')}_40ms_sensores.png")
        plt.savefig(save_file)
        plt.show()
        print(f"{class_name} sınıfı için tüm sensörlerin figürü {save_file} konumuna kaydedildi.")


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

    def plot_class_comparison(self):
        """
        Her sınıf için sensör verilerinin ortalamalarını karşılaştırır.
        """
        if self.dataset is None:
            raise ValueError("Veri seti yüklenmemiş. Önce `load_data` çağrılmalı.")

        save_path = "results/dataset"
        os.makedirs(save_path, exist_ok=True)

        mean_values = []
        for gesture_class in self.dataset["Gesture_Class"].unique():
            class_data = self.dataset[self.dataset["Gesture_Class"] == gesture_class].iloc[:, :-1]
            mean_values.append(class_data.mean(axis=0).values)

        mean_values = np.array(mean_values)

        plt.figure(figsize=(12, 6))
        for i, class_name in enumerate(self.class_names):
            plt.plot(range(1, mean_values.shape[1] + 1), mean_values[i], label=class_name)

        plt.title("Sınıf Bazında Ortalama Sensör Verileri")
        plt.xlabel("Özellik İndeksi")
        plt.ylabel("Ortalama Değer")
        plt.legend(loc="upper right")
        plt.grid(True)

        save_file = os.path.join(save_path, "class_comparison.png")
        plt.savefig(save_file)
        plt.show()
        print(f"Sınıf karşılaştırma grafiği {save_file} konumuna kaydedildi.")

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

    # Sınıfları karşılaştır
    processor.plot_class_comparison()

    print("\n--- Tüm görselleştirme işlemleri tamamlandı ---")
