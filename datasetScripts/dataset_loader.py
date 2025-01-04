import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

class DatasetLoader:
    def __init__(self, file_path):
        """
        DatasetLoader sınıfı, bir veri dosyasını yüklemek ve analiz etmek için kullanılır.
        :param file_path: Veri dosyasının yolu
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Veri dosyasını yükler ve DataFrame'e aktarır."""
        try:
            print(f"Veri yükleniyor: {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            print(f"Veri başarıyla yüklendi. Toplam satır sayısı: {len(self.data)}")
        except Exception as e:
            print(f"Veri yüklenirken bir hata oluştu: {e}")
            self.data = None

    def summarize_data(self):
        """Verinin genel özet istatistiklerini ve sütun bilgilerini verir."""
        if self.data is not None:
            print("\n=== Veri Özet Bilgisi ===")
            print(self.data.info())
            print("\n=== İlk 5 Satır ===")
            print(self.data.head())
            print("\n=== Eksik Değerler ===")
            print(self.data.isnull().sum())
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def remove_unmarked_data(self, class_column="class"):
        """
        İşaretlenmemiş veriyi (sınıf 0) veri setinden çıkarır.
        :param class_column: Sınıf sütununun adı
        """
        if self.data is not None:
            original_size = len(self.data)
            self.data = self.data[self.data[class_column] != 0]
            print(f"Sınıf 0 verileri çıkarıldı. Kalan veri sayısı: {len(self.data)} (Önceki: {original_size})")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")

    def plot_class_distribution(self, class_column="class"):
        """
        Veri setindeki sınıf dağılımını çizer ve parametrelerle damgalı şekilde kaydeder.
        :param class_column: Sınıf sütununun adı
        """
        if self.data is not None:
            if class_column in self.data.columns:
                print(f"\nSınıf dağılımı görselleştiriliyor: {class_column}")
                class_counts = self.data[class_column].value_counts()

                plt.figure(figsize=(10, 6))
                bars = class_counts.plot(kind='bar', color='skyblue', alpha=0.8)
                plt.title("Sınıf Dağılımı")
                plt.xlabel("Sınıf")
                plt.ylabel("Örnek Sayısı")
                plt.xticks(rotation=45)

                for idx, count in enumerate(class_counts):
                    plt.text(idx, count + (0.01 * count), f'{count}', ha='center', va='bottom', fontsize=10)

                # Plot dosya adı için parametreleri damgala
                plots_dir = "plots"
                os.makedirs(plots_dir, exist_ok=True)

                class_counts_str = "_".join([f"{cls}_{cnt}" for cls, cnt in class_counts.items()])
                plot_filename = f"class_distribution_{class_counts_str}.png"

                plot_path = os.path.join(plots_dir, plot_filename)
                plt.savefig(plot_path)
                plt.close()  # Grafiği kapat
                print(f"Grafik kaydedildi: {plot_path}")
            else:
                print(f"{class_column} sütunu veri setinde bulunamadı.")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")



    def plot_class_distribution_pca(self, class_column="class"):
        """Sınıfların PCA ile 2D uzaydaki dağılımını çizer."""
        if self.data is not None:
            if class_column in self.data.columns:
                print("\nSınıfların uzaydaki dağılımı PCA ile görselleştiriliyor.")
                
                # Sınıf sütunu hariç diğer sütunlar
                feature_columns = [col for col in self.data.columns if col not in [class_column, 'Time', 'label']]
                
                # PCA uygulanabilir mi kontrol edelim
                if len(feature_columns) > 1:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(self.data[feature_columns])
                    
                    # PCA sonuçlarını DataFrame'e ekle
                    self.data['PCA1'] = pca_result[:, 0]
                    self.data['PCA2'] = pca_result[:, 1]
                    
                    plt.figure(figsize=(12, 8))
                    for class_label in self.data[class_column].unique():
                        subset = self.data[self.data[class_column] == class_label]
                        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Sınıf {class_label}", alpha=0.6)
                    
                    plt.title("Sınıfların Uzaydaki Dağılımı (PCA)")
                    plt.xlabel("PCA1")
                    plt.ylabel("PCA2")
                    plt.legend()
                    plt.grid(True)
                    
                    pca_plot_dir = "plots/pca"
                    os.makedirs(pca_plot_dir, exist_ok=True)
                    plt.savefig(f"{pca_plot_dir}/class_distribution_pca.png")
                    plt.close()
                    print("PCA görselleştirmesi kaydedildi.")
                else:
                    print("PCA uygulanabilir değil. Yeterli sayıda sayısal sütun yok.")
            else:
                print(f"{class_column} sütunu veri setinde bulunamadı.")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")



    def plot_class_distribution_tsne(self, class_column="class", samples_per_class=1000):
        """
        Sadece channel sütunlarını kullanarak t-SNE ile sınıfların uzaydaki dağılımını görselleştirir.
        :param class_column: Sınıf sütununun adı
        :param samples_per_class: Her sınıftan alınacak örnek sayısı
        """
        if self.data is not None:
            if class_column in self.data.columns:
                print("\nSınıfların uzaydaki dağılımı t-SNE ile görselleştiriliyor.")
                
                # Sadece channel sütunlarını seçiyoruz
                channel_columns = ["channel1", "channel2", "channel3", "channel4", 
                                "channel5", "channel6", "channel7", "channel8"]
                
                if all(col in self.data.columns for col in channel_columns):
                    # Her sınıftan belirtilen sayıda örnek seçme
                    sampled_data = self.data.groupby(class_column).apply(
                        lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
                    ).reset_index(drop=True)
                    
                    # t-SNE uygulama
                    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                    tsne_result = tsne.fit_transform(sampled_data[channel_columns])
                    
                    sampled_data['tSNE1'] = tsne_result[:, 0]
                    sampled_data['tSNE2'] = tsne_result[:, 1]
                    
                    # Renk paletini oluşturma
                    cmap = plt.get_cmap("tab10")  # "tab10" paleti genelde 10 ayırt edici renk sağlar
                    colors = [cmap(i) for i in range(8)]
                    
                    # Görselleştirme
                    plt.figure(figsize=(12, 8))
                    for i, class_label in enumerate(sampled_data[class_column].unique()):
                        subset = sampled_data[sampled_data[class_column] == class_label]
                        plt.scatter(
                            subset['tSNE1'], 
                            subset['tSNE2'], 
                            label=f"Sınıf {class_label}", 
                            alpha=0.6, 
                            color=colors[i]
                        )
                    
                    plt.title("Sınıfların Uzaydaki Dağılımı (t-SNE, Channel Sütunları)")
                    plt.xlabel("tSNE1")
                    plt.ylabel("tSNE2")
                    plt.legend()
                    plt.grid(True)
                    
                    tsne_plot_dir = "plots/tsne"
                    os.makedirs(tsne_plot_dir, exist_ok=True)
                    plt.savefig(f"{tsne_plot_dir}/class_distribution_tsne_channels.png")
                    plt.close()
                    print("t-SNE görselleştirmesi kaydedildi.")
                else:
                    print("Channel sütunları eksik. Lütfen veri setinizi kontrol edin.")
            else:
                print(f"{class_column} sütunu veri setinde bulunamadı.")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")



    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    import os

    def plot_decision_tree(self, class_column="class", output_dir="plots/decision_tree"):
        """
        Karar ağacını görselleştirir ve bir dosyaya kaydeder.
        :param class_column: Sınıf sütununun adı
        :param output_dir: Çıktı dosyasının kaydedileceği dizin
        """
        if self.data is not None:
            feature_columns = ["channel1", "channel2", "channel3", "channel4", 
                            "channel5", "channel6", "channel7", "channel8"]

            if all(col in self.data.columns for col in feature_columns):
                X = self.data[feature_columns]
                y = self.data[class_column]

                # Karar ağacı modeli
                clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                clf.fit(X, y)

                # Kaydetme için çıktı dizinini oluştur
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "decision_tree.png")

                # Karar ağacını görselleştir
                plt.figure(figsize=(20, 10))
                plot_tree(clf, feature_names=feature_columns, class_names=[str(c) for c in clf.classes_], filled=True)
                plt.title("Karar Ağacı Görselleştirmesi")
                
                # Görselleştirmeyi dosyaya kaydet
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()  # Grafiği kapat
                print(f"Karar ağacı kaydedildi: {output_file}")
            else:
                print("Channel sütunları eksik. Lütfen veri setinizi kontrol edin.")
        else:
            print("Veri yüklenmemiş. Önce load_data() çağrılmalı.")




# Eğer bu dosya çalıştırılırsa
if __name__ == "__main__":
    # Veri dosyasının yolu
    dataset_path = "../dataset/EMG-data.csv"

    loader = DatasetLoader(dataset_path)

    # Veriyi yükle
    loader.load_data()

    # Veriyi özetle
    loader.summarize_data()

    # # Sınıf 0'ı çıkar
    loader.remove_unmarked_data()

    # Sınıf dağılımını görselleştir
    loader.plot_class_distribution_tsne(class_column="class",samples_per_class=1000)
    loader.plot_decision_tree(class_column="class")
    loader.plot_class_distribution(class_column="class")
