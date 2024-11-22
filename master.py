from dataset import DataProcessor
from evaluation import ModelEvaluator
from random_forest_model import RandomForestClassifierModel
import os
import numpy as np

# Sınıflar
classes = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]

# Verinin yüklenmesi ve işlenmesi
data_path = "dataset/"
processor = DataProcessor(data_path=data_path, class_names=classes)
dataset = processor.load_data()

# Özellik ve hedef ayrımı
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = processor.train_test_split(X, y)

# Eğitim ve test seti dağılım kontrolü
processor.check_train_test_distribution(y_train, y_test)

# Random Forest Modeli
def train_and_evaluate_random_forest():
    print("\n--- Random Forest Modeli Eğitimi ve Değerlendirilmesi ---\n")
    rf_model = RandomForestClassifierModel(n_estimators=100)
    y_pred_rf = rf_model.train_and_predict(X_train, y_train, X_test)

    # Değerlendirme
    evaluator = ModelEvaluator(class_names=classes)
    evaluator.evaluate(y_test, y_pred_rf, model_name="Random Forest")

if __name__ == "__main__":
    # Çıkış klasörlerini oluştur
    os.makedirs("results", exist_ok=True)

    # Random Forest Modeli
    train_and_evaluate_random_forest()

    print("\n--- Tüm İşlemler Tamamlandı ---")
