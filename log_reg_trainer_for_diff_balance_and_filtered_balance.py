import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionTrainer:
    def __init__(self, data, target_column):
        """
        Logistic Regression modeli eğitmek ve değerlendirmek için sınıf.
        :param data: Veri seti (pandas DataFrame)
        :param target_column: Sınıf etiketlerini içeren sütun adı
        """
        self.data = data
        self.target_column = target_column

    def preprocess_data(self):
        """
        Veriyi train, validation ve test setlerine böler.
        :return: Train, validation ve test setleri
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Veri setini train, validation ve test setlerine böl
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print(f"Train set boyutu: {X_train.shape[0]} örnek")
        print(f"Validation set boyutu: {X_val.shape[0]} örnek")
        print(f"Test set boyutu: {X_test.shape[0]} örnek")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Logistic Regression modelini eğitir ve performansını değerlendirir.
        """
        # Özellikleri ölçekle
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression modelini oluştur
        model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        model.fit(X_train_scaled, y_train)

        # Validation seti üzerinde değerlendirme
        val_predictions = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Test seti üzerinde değerlendirme
        test_predictions = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print("\nValidation Set Sonuçları:")
        print(f"Doğruluk (Accuracy): {val_accuracy:.2f}")
        print(classification_report(y_val, val_predictions))

        print("\nTest Set Sonuçları:")
        print(f"Doğruluk (Accuracy): {test_accuracy:.2f}")
        print(classification_report(y_test, test_predictions))

        # Confusion matrix hesapla
        cm = confusion_matrix(y_test, test_predictions)

        # Başarı özetini döndür
        return {
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "confusion_matrix": cm
        }

    @staticmethod
    def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
        """
        Confusion matrix'i görselleştirir.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=16)
        plt.xlabel("Tahmin Edilen Sınıf", fontsize=12)
        plt.ylabel("Gerçek Sınıf", fontsize=12)
        plt.show()


if __name__ == "__main__":
    # Dengelenmiş ve dengelenmemiş veri setlerini yükle
    filtered_data = pd.read_csv("./dataset/filtered_balanced_emg_data.csv")
    balanced_data = pd.read_csv("./dataset/Balanced_EMG_data.csv")

    # LogisticRegressionTrainer sınıflarını başlat
    print("Dengelenmemiş Veri Seti ile Eğitim ve Değerlendirme:")
    trainer_original = LogisticRegressionTrainer(filtered_data, target_column="class")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer_original.preprocess_data()
    results_original = trainer_original.train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\nDengelenmiş Veri Seti ile Eğitim ve Değerlendirme:")
    trainer_balanced = LogisticRegressionTrainer(balanced_data, target_column="class")
    X_train_bal, X_val_bal, X_test_bal, y_train_bal, y_val_bal, y_test_bal = trainer_balanced.preprocess_data()
    results_balanced = trainer_balanced.train_and_evaluate(X_train_bal, X_val_bal, X_test_bal, y_train_bal, y_val_bal, y_test_bal)

    # Başarıları özetle ve yazdır
    print("\n--- SONUÇLAR ---")
    print("dengelenmiş filtrelenmiş Veri Seti:")
    print(f"Validation Accuracy: {results_original['val_accuracy']:.2f}")
    print(f"Test Accuracy: {results_original['test_accuracy']:.2f}")
    print("Confusion Matrix (dengelenmiş filtrelenmiş):")
    trainer_original.plot_confusion_matrix(results_original['confusion_matrix'], classes=np.unique(y_test), title="Confusion Matrix (dengelenmiş filtrelenmiş)")

    print("\nDengelenmiş Veri Seti:")
    print(f"Validation Accuracy: {results_balanced['val_accuracy']:.2f}")
    print(f"Test Accuracy: {results_balanced['test_accuracy']:.2f}")
    print("Confusion Matrix (Dengelenmiş):")
    trainer_balanced.plot_confusion_matrix(results_balanced['confusion_matrix'], classes=np.unique(y_test_bal), title="Confusion Matrix (Dengelenmiş)")
