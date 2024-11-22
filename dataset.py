import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    def preprocess(self):
        """
        Veriyi normalize eder ve şekillendirir.
        :return: Normalized and reshaped data (X), labels (y)
        """
        print("Veri işleniyor...")
        X = np.array(self.dataset.iloc[:, :-1])  # Özellikler
        y = np.array(self.dataset.iloc[:, -1])  # Etiketler

        # Veriyi normalize et
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Veriyi yeniden şekillendir (8x8 matris)
        X = X.reshape(-1, 8, 8)
        for i in range(len(X)):
            X[i] = X[i].T  # Transpose işlemi

        print("Veri normalleştirildi ve yeniden şekillendirildi.\n")
        return X, y

    def train_test_split(self, X, y, test_size=0.33):
        """Veriyi eğitim ve test setlerine böler."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        print("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    
    def check_train_test_distribution(self, y_train, y_test, sequential=False):
        """
        Eğitim ve test setindeki sınıf dağılımlarını gözlemler ve çıktı verir.

        :param y_train: Eğitim setindeki etiketler
        :param y_test: Test setindeki etiketler
        :param sequential: Etiketlerin one-hot encoding formatında olup olmadığını belirtir
        """
        # One-hot encoding varsa, decoder ile düzelt
        if sequential:
            y_train = self.decoder(y_train)
            y_test = self.decoder(y_test)

        # Eğitim ve test seti için sınıf sayımlarını hesapla
        train_class_counts = [(y_train == i).sum() for i in range(len(self.class_names))]
        test_class_counts = [(y_test == i).sum() for i in range(len(self.class_names))]

        # Dağılımları yazdır
        width_x = max(len(x) for x in self.class_names)
        res = "\n".join(
            "{:>{}} : {}  {}".format(cls, width_x, train_count, test_count)
            for cls, train_count, test_count in zip(self.class_names, train_class_counts, test_class_counts)
        )
        print("Eğitim ve test seti sınıf dağılımları:")
        print(res + "\n")

    def decoder(self, y_list):
        """
        One-hot encoded etiketleri düz sınıf etiketlerine çevirir.
        """
        return np.array([np.argmax(y) for y in y_list])

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
        print(f"Figür {save_file} konumuna kaydedildi.")

if __name__ == '__main__':
    data_path = "dataset/"
    classes = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

    # DataProcessor nesnesi oluşturuluyor
    processor = DataProcessor(data_path, classes)

    # Veriyi yükle
    dataset = processor.load_data()

    # Veriyi işle (preprocess)
    X, y = processor.preprocess()

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

    # Eğitim ve test dağılımını kontrol et
    processor.check_train_test_distribution(y_train, y_test)

    # Eğitim ve test seti dağılımlarını bir dosyaya kaydet
    results_dir = "results/dataset"
    os.makedirs(results_dir, exist_ok=True)

    txt_file_path = os.path.join(results_dir, "train_test_distribution.txt")
    with open(txt_file_path, "w") as f:
        f.write("Eğitim ve Test Sınıf Dağılımı:\n")
        train_class_counts = [(y_train == i).sum() for i in range(len(classes))]
        test_class_counts = [(y_test == i).sum() for i in range(len(classes))]
        width_x = max(len(x) for x in classes)
        for cls, train_count, test_count in zip(classes, train_class_counts, test_class_counts):
            line = f"{cls:>{width_x}}(Sınıf) : {train_count} (Eğitim Data Sayısı) {test_count} (Test Data Sayısı)\n"
            f.write(line)
        print(f"\nEğitim ve test seti dağılımı '{txt_file_path}' dosyasına kaydedildi.")

    # Her sınıfın ilk satırını görselleştir
    print("\n--- Görselleştirme: Sensör Verisi ---\n")
    for i, class_name in enumerate(classes):
        data_row = processor.dataset.iloc[i]
        print(f"Görselleştiriliyor: {class_name}")
        processor.plot_8sensors_data(data_row, title=class_name)

    print("\n--- Tüm İşlemler Tamamlandı ---")
