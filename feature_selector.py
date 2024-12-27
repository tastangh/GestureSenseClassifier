import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import os


class FeatureSelector:
    """
    Özellik seçimi için kullanılan sınıf.
    """
    def __init__(self, features_df, target_column="Label", output_dir="results/feature_selection"):
        """
        :param features_df: Özellik matrisi (DataFrame)
        :param target_column: Hedef sütunun adı
        :param output_dir: Sonuçların kaydedileceği dizin
        """
        self.features_df = features_df
        self.target_column = target_column
        self.output_dir = output_dir
        self.X = self.features_df.drop(columns=[self.target_column])
        self.y = self.features_df[self.target_column]
        os.makedirs(self.output_dir, exist_ok=True)

    def remove_highly_correlated_features(self, threshold=0.9):
        """
        Yüksek korelasyona sahip özellikleri kaldırır.
        :param threshold: Korelasyon eşiği
        :return: Kalan özelliklerin DataFrame'i
        """
        correlation_matrix = self.X.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]
        print(f"Kaldırılacak yüksek korelasyonlu özellikler: {to_drop}")
        self.X = self.X.drop(columns=to_drop)
        return self.X

    def select_top_k_features(self, k=10, method="f_classif"):
        """
        En iyi k özelliği seçer.
        :param k: Seçilecek özellik sayısı
        :param method: Özellik seçimi yöntemi ('f_classif', 'mutual_info')
        :return: Seçilen özelliklerin DataFrame'i
        """
        if method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError("Geçersiz yöntem: 'f_classif' veya 'mutual_info' kullanın.")

        selector.fit(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()]
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            "Feature": self.X.columns,
            "Score": scores
        }).sort_values(by="Score", ascending=False)

        # Seçilen özelliklerin görselleştirilmesi
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Score", y="Feature", data=feature_scores.head(k), palette="viridis")
        plt.title(f"Top {k} Features ({method})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"top_{k}_features_{method}.png"))
        plt.close()

        print(f"Seçilen en iyi {k} özellik: {selected_features.tolist()}")
        self.X = self.X[selected_features]
        return self.X

    def model_based_feature_selection(self, model=None, top_n=10):
        """
        Model tabanlı özellik seçimi.
        :param model: Kullanılacak model (örneğin, RandomForestClassifier)
        :param top_n: En önemli n özelliği seç
        :return: Seçilen özelliklerin DataFrame'i
        """
        if model is None:
            model = RandomForestClassifier(random_state=42)

        model.fit(self.X, self.y)
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Özellik önemlerinin görselleştirilmesi
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_importances.head(top_n), palette="magma")
        plt.title(f"Top {top_n} Features (Model-Based)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_based_top_features.png"))
        plt.close()

        selected_features = feature_importances.head(top_n)["Feature"].tolist()
        print(f"Model tabanlı seçilen özellikler: {selected_features}")
        self.X = self.X[selected_features]
        return self.X


if __name__ == "__main__":
    # Örnek veri yükleme
    raw_features_path = "results/features/raw_features_with_labels.csv"
    filtered_features_path = "results/features/filtered_features_with_labels.csv"

    raw_features_df = pd.read_csv(raw_features_path)
    filtered_features_df = pd.read_csv(filtered_features_path)

    # Ham özellikler üzerinde seçim
    print("Ham özellikler üzerinde seçim yapılıyor...")
    raw_selector = FeatureSelector(raw_features_df)
    raw_selector.remove_highly_correlated_features()
    raw_selector.select_top_k_features(k=10, method="f_classif")
    raw_selector.model_based_feature_selection(top_n=10)

    # Filtrelenmiş özellikler üzerinde seçim
    print("Filtrelenmiş özellikler üzerinde seçim yapılıyor...")
    filtered_selector = FeatureSelector(filtered_features_df)
    filtered_selector.remove_highly_correlated_features()
    filtered_selector.select_top_k_features(k=10, method="mutual_info")
    filtered_selector.model_based_feature_selection(top_n=10)

    print("\n--- Tüm Özellik Seçimi İşlemleri Tamamlandı ---")
