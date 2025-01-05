from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import os 

class DatasetBalancer:
    def __init__(self, target_size=250000, random_state=42):
        """
        DatasetBalancer sınıfı, oversampling kullanarak sınıf dengesini sağlar.
        :param target_size: Her sınıf için hedef boyut
        :param random_state: Rastgelelik kontrolü için seed
        """
        self.target_size = target_size
        self.random_state = random_state
        self.smote = SMOTE(random_state=self.random_state)

    def balance_data(self, data, class_column="class"):
        """
        Veriyi SMOTE kullanarak dengeler.
        :param data: Veri seti (pandas DataFrame)
        :param class_column: Sınıf sütununun adı
        :return: Dengelenmiş veri seti (pandas DataFrame)
        """
        label_counts = data[class_column].value_counts()

        print("\n=== Başlangıç Sınıf Dağılımı ===")
        print(label_counts)

        # Özellik ve etiketleri ayır
        X = data.drop(columns=[class_column])
        y = data[class_column]

        # Eğer herhangi bir sınıfın örnek sayısı çok az ise SMOTE uygulanamaz
        if label_counts.min() < 2:
            print("Bazı sınıflar SMOTE için yetersiz örneğe sahip.")
            print("Önce random oversampling yapılacak...")

            # Çok az örneği olan sınıflar için random oversampling
            min_count = label_counts.min()
            if min_count < 2:
                for label, count in label_counts.items():
                    if count < 2:
                        class_data = data[data[class_column] == label]
                        oversampled_data = class_data.sample(
                            n=2, replace=True, random_state=self.random_state
                        )
                        data = pd.concat([data, oversampled_data], axis=0)

                # Güncellenen sınıf dağılımını yazdır
                label_counts = data[class_column].value_counts()
                print("\n=== Güncellenmiş Sınıf Dağılımı (Random Oversampling Sonrası) ===")
                print(label_counts)

            # Özellik ve etiketleri yeniden ayır
            X = data.drop(columns=[class_column])
            y = data[class_column]

        # SMOTE ile dengeleme
        X_balanced, y_balanced = self.smote.fit_resample(X, y)

        # Dengelenmiş veriyi birleştir
        balanced_data = pd.concat(
            [pd.DataFrame(X_balanced), pd.Series(y_balanced, name=class_column)], axis=1
        )

        # Hedef boyuta göre sınırlama
        final_data = balanced_data.groupby(class_column).apply(
            lambda group: group.sample(n=self.target_size, random_state=self.random_state)
        ).reset_index(drop=True)

        return final_data

    def plot_class_distribution(self, data, class_column="class", title="Dengelenmiş Sınıf Dağılımı"):
        """
        Dengelenmiş sınıf dağılımını görselleştirir.
        :param data: Dengelenmiş veri seti
        :param class_column: Sınıf sütununun adı
        :param title: Grafik başlığı
        """
        class_counts = data[class_column].value_counts()
        print("\n=== Dengelenmiş Sınıf Dağılımı ===")
        print(class_counts)

        plt.figure(figsize=(10, 6))
        bars = class_counts.plot(kind="bar", color="green", alpha=0.8)
        plt.title(title)
        plt.xlabel("Sınıf")
        plt.ylabel("Örnek Sayısı")
        plt.xticks(rotation=45)

        for idx, count in enumerate(class_counts):
            plt.text(idx, count + (0.01 * count), f"{count}", ha="center", va="bottom", fontsize=10)

        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, f"class_balanced_distribution_{self.target_size}_samples.png")
        plt.savefig(plot_path)
        print(f"Grafik kaydedildi: {plot_path}")


if __name__ == "__main__":
    from dataset_loader import DatasetLoader

    # Dengelenmiş veri dosyasını oluştur
    dataset_path = "dataset/EMG-data.csv"
    loader = DatasetLoader(dataset_path)

    # Veriyi yükle ve 0 sınıfını kaldır
    loader.load_data()
    loader.remove_unmarked_data()

    # Veriyi dengele
    balancer = DatasetBalancer(target_size=250000)
    balanced_data = balancer.balance_data(loader.data)

    # Dengelenmiş sınıf dağılımını görselleştir
    balancer.plot_class_distribution(balanced_data, class_column="class")

    # Dengelenmiş veriyi kaydet
    output_dir = "dataset"
    output_path = f"{output_dir}/balanced_EMG_data.csv"
    balanced_data.to_csv(output_path, index=False)
    print(f"Dengelenmiş veri seti kaydedildi: {output_path}")
