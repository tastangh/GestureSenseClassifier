from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np 

class RandomForestTrainer:
    """
    Random Forest modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, class_weight=None):
        """
        RandomForestTrainer sınıfını başlatır.
        :param n_estimators: Ağaç sayısı
        :param max_depth: Maksimum derinlik
        :param random_state: Rastgelelik kontrolü
        :param class_weight: Sınıf ağırlıkları ('balanced' veya None)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Modeli eğitir ve metrikleri hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        """
        unique_classes = np.unique(y_train)

        # Modeli eğit
        self.model.fit(X_train, y_train)

        # Eğitim metriklerini hesapla
        y_train_pred_prob = self.model.predict_proba(X_train)
        self.train_loss = log_loss(y_train, y_train_pred_prob, labels=unique_classes)
        self.train_accuracy = accuracy_score(y_train, self.model.predict(X_train))

        # Doğrulama metriklerini hesapla
        if X_val is not None and y_val is not None:
            y_val_pred_prob = self.model.predict_proba(X_val)
            self.val_loss = log_loss(y_val, y_val_pred_prob, labels=unique_classes)
            self.val_accuracy = accuracy_score(y_val, self.model.predict(X_val))

        # Loglama
        print("\nEğitim Tamamlandı!")
        print(f"Eğitim Kaybı: {self.train_loss:.4f} | Eğitim Doğruluk: {self.train_accuracy:.4f}")
        if self.val_loss is not None:
            print(f"Doğrulama Kaybı: {self.val_loss:.4f} | Doğrulama Doğruluk: {self.val_accuracy:.4f}")

    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        return self.model.predict(X_test)
