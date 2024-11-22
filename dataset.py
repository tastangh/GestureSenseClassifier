import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path, class_names, num_features=64):
        """
        :param data_path: Veri dosyalarının bulunduğu dizin
        :param class_names: Sınıf isimlerinin listesi
        :param num_features: Her veri noktasının sahip olduğu özellik sayısı
        """
        self.data_path = data_path
        self.class_names = class_names
        self.num_features = num_features
        self.dataset = None

    def load_data(self):
        """Tüm sınıflar için veri dosyalarını yükler ve birleştirir."""
        frames = []
        for i, class_name in enumerate(self.class_names):
            file_path = f"{self.data_path}{i}.csv"
            print(f"Loading {file_path}...")
            data = pd.read_csv(file_path, header=None)
            frames.append(data)
        self.dataset = pd.concat(frames)
        print("All data loaded and concatenated.")
        return self.dataset

    def train_test_split(self, X, y, test_size=0.33):
        """Veriyi eğitim ve test setlerine böler."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test

    def plot_8sensors_data(self, data_row, title, interval=40, no_of_sensors=8, n_steps=8):
        """
        8 sensör verisini görselleştirmek için kullanılır.

        :param data_row: Görselleştirilmek istenen satır (örnek olarak bir veri noktası)
        :param title: Grafiğin başlığı
        :param interval: Zaman aralığı
        :param no_of_sensors: Sensör sayısı
        :param n_steps: Her sensör için zaman adımı
        """
        # Klasör yolunu oluştur
        save_path = "results/dataset"
        os.makedirs(save_path, exist_ok=True)

        # Türkçe başlık düzenlemesi
        class_label = title.split("(")[0].strip()
        class_number = title.split("(")[1].replace(")", "")  # "1"
        title = f"{class_label} Hareketi Sınıfı ({class_number})"  

        # Dosya ismi düzenleme
        file_name = f"{class_label.lower()}_{class_number}_class.png"

        data = data_row.values.flatten()  # Pandas serisini numpy array'e çeviriyoruz
        xTime = np.linspace(interval, interval * n_steps, n_steps)
        yInterval = np.linspace(-128, 128, 1)

        n = 1
        fig = plt.figure()
        for i in range(0, len(data)-1, n_steps):
            plt.subplot(int(no_of_sensors/2), 2, n)
            plt.plot(xTime, data[i: i + n_steps])
            plt.title(f"{n}. sensor")
            plt.xlabel("zaman(ms)")
            plt.ylabel("örnekler")
            plt.xticks(xTime)
            plt.yticks(yInterval)
            n += 1

        plt.suptitle(title)
        plt.tight_layout()

        # Görseli kaydet
        save_file = os.path.join(save_path, file_name)
        fig.savefig(save_file, dpi=100)
        plt.show()
        print(f"Figure saved to {save_file}")


# Eğer dosya bağımsız çalıştırılırsa
if __name__ == '__main__':

    data_path = "dataset/"
    classes = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]


    # DataProcessor nesnesi oluşturuluyor
    processor = DataProcessor(data_path, classes)

    # Veriyi yükle
    processor.load_data()

    # Her sınıfın ilk satırını görselleştir
    print("Visualizing sensor data for each class...")
    for i, class_name in enumerate(classes):
        data_row = processor.dataset.iloc[i]
        processor.plot_8sensors_data(data_row, title=class_name)


    print("Visualization completed.")
