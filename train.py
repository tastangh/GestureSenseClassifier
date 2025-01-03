import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Model kaydetmek için
from dataset_processor import DatasetProcessor # DatasetProcessor'ı import et

class Trainer:
    def __init__(self, dataset_path, output_dir="models", test_size=0.2, val_size=0.2, random_state=42):
        """
        Model eğitimini yönetir.
        :param dataset_path: Veri setinin yolu
        :param output_dir: Model ve çıktılar için klasör
        :param test_size: Test kümesinin oranı
        :param val_size: Validasyon kümesinin oranı
        :param random_state: Rastgelelik için seed değeri
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.model = None
        self.processor = None
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self):
       """Veri setini hazırlar ve böler."""
       print("Veri hazırlanıyor...")
       self.processor = DatasetProcessor(data_path=self.dataset_path, test_size=self.test_size, val_size=self.val_size, random_state=self.random_state, balance_data=True, apply_filter=True, extract_features=True, normalize_data=True)
       self.processor.process(class_column="class", output_dir=self.output_dir)
       print("Veri hazırlandı.")


    def train_model(self, class_column="class"):
        """Lojistik regresyon modelini eğitir ve en iyi parametreleri bulur."""
        print("Model eğitiliyor...")

        if self.processor is None:
            raise ValueError("Veri hazırlanmamış. Önce prepare_data() çağrılmalı.")

        if self.processor.train_data is None:
            raise ValueError("Eğitim verisi bulunamadı. Lütfen veriyi bölme işlemini kontrol edin.")


        train_data = self.processor.train_data
        X_train = train_data.drop(columns=[class_column])
        y_train = train_data[class_column]

        # Hiperparametreleri tanımla
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # GridSearchCV ile hiperparametre optimizasyonu
        grid_search = GridSearchCV(LogisticRegression(random_state=self.random_state, max_iter=1000), 
                                   param_grid, 
                                   cv=5, 
                                   scoring='accuracy', 
                                   verbose=1, 
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # En iyi modeli ve parametreleri al
        self.model = grid_search.best_estimator_
        print("En iyi parametreler:", grid_search.best_params_)
        print("Model başarıyla eğitildi.")


    def evaluate_model(self, class_column="class"):
        """Modeli değerlendirir ve sonuçları yazdırır."""
        print("Model değerlendiriliyor...")

        if self.model is None:
           raise ValueError("Model eğitilmemiş. Önce train_model() çağrılmalı.")
        
        if self.processor is None:
             raise ValueError("Veri hazırlanmamış. Önce prepare_data() çağrılmalı.")

        if self.processor.test_data is None:
              raise ValueError("Test verisi bulunamadı. Lütfen veriyi bölme işlemini kontrol edin.")

        test_data = self.processor.test_data
        X_test = test_data.drop(columns=[class_column])
        y_test = test_data[class_column]

        # Test verisi üzerinde tahmin yap
        y_pred = self.model.predict(X_test)

        # Performansı değerlendir
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test doğruluğu: {accuracy:.4f}")
        print("Sınıflandırma raporu:\n", classification_report(y_test, y_pred))


    def save_model(self, model_name="logistic_regression_model.pkl"):
        """Eğitilmiş modeli kaydeder."""
        if self.model is not None:
            model_path = os.path.join(self.output_dir, model_name)
            joblib.dump(self.model, model_path)
            print(f"Model kaydedildi: {model_path}")
        else:
            print("Kaydedilecek model bulunamadı.")


    def run(self, class_column="class"):
        """Tüm eğitim sürecini yönetir."""
        self.prepare_data()
        self.train_model(class_column=class_column)
        self.evaluate_model(class_column=class_column)
        self.save_model()


if __name__ == "__main__":
    # Veri setinin yolu
    dataset_path = "dataset/EMG-data.csv"
    output_dir = "models"
    # Trainer nesnesi oluştur
    trainer = Trainer(dataset_path=dataset_path, output_dir=output_dir, test_size=0.2, val_size=0.2)

    # Tüm süreci başlat
    trainer.run(class_column="class")