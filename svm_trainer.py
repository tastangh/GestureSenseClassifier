from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMTrainer:
    """
    SVM modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, kernel="linear", C=1.0, random_state=42):
        """
        SVMTrainer sınıfını başlatır.
        :param kernel: Çekirdek türü ('linear', 'rbf', 'poly', vb.)
        :param C: Ceza parametresi
        :param random_state: Rastgelelik kontrolü
        """
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
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
        # Modeli eğit
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
