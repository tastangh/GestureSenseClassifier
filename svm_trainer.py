from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class SVMTrainer:
    """
    SVM modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self,  random_state=42):
        """
        SVMTrainer sınıfını başlatır.
        :param kernel: Çekirdek türü ('linear', 'rbf', 'poly', vb.)
        :param C: Ceza parametresi
        :param random_state: Rastgelelik kontrolü
        """
        self.model = None
        self.random_state = random_state
        self.train_accuracy = None
        self.val_accuracy = None

    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, n_trials=10):
         # Random search için hiperparametre aralığı
        param_dist = {
            "C": np.logspace(-4, 4, 20),  # C için logaritmik aralık
            "kernel": ['linear', 'rbf', 'poly']  # kernel için seçenekler
        }
        
        # RandomizedSearchCV ile optimizasyon
        svm_random = RandomizedSearchCV(
            SVC(random_state=self.random_state, probability=True, class_weight="balanced"),
            param_distributions=param_dist,
            n_iter=n_trials,  # Deneme sayısı
            cv=3,  # Cross validation sayısı
            random_state=self.random_state,
            scoring="accuracy",
        )
        
        svm_random.fit(X_train, y_train)
        
        print("SVM - En iyi Parametreler:", svm_random.best_params_)
        self.model = svm_random.best_estimator_
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=True, n_trials=10):
        """
        Modeli eğitir ve metrikleri hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        """
        if optimize:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
        else:
            self.model = SVC(kernel="linear", C=1.0, probability=True, random_state=self.random_state, class_weight="balanced")
            self.model.fit(X_train, y_train)

        # Eğitim doğruluğunu hesapla
        self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))

        # Doğrulama doğruluğunu hesapla
        if X_val is not None and y_val is not None:
            self.val_accuracy = accuracy_score(y_val, self.model.predict(X_val))

        # Loglama
        print("\nEğitim Tamamlandı!")
        print(f"Eğitim Doğruluk: {self.train_accuracy:.4f}")
        if self.val_accuracy is not None:
            print(f"Doğrulama Doğruluk: {self.val_accuracy:.4f}")

    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        return self.model.predict(X_test)