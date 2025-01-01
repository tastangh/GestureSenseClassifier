import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

class ANNTrainer:
    """
    ANN modelini eğitmek ve metrikleri hesaplamak için sınıf.
    """
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, learning_rate=0.001):
        """
        ANNTrainer sınıfını başlatır.
        :param input_dim: Giriş boyutu (özellik sayısı)
        :param hidden_layers: Her bir gizli katmandaki nöron sayısını içeren liste
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        """
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Modeli eğitir ve eğitim/doğrulama metriklerini hesaplar.
        :param X_train: Eğitim verisi
        :param y_train: Eğitim etiketi
        :param X_val: Doğrulama verisi (isteğe bağlı)
        :param y_val: Doğrulama etiketi (isteğe bağlı)
        :param epochs: Eğitim için epoch sayısı
        :param batch_size: Eğitim için batch boyutu
        """
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        print("\nEğitim Tamamlandı!")

    def predict(self, X_test):
        """
        Test verisi üzerinde tahmin yapar.
        :param X_test: Test verisi
        :return: Tahmin edilen etiketler
        """
        predictions = self.model.predict(X_test)
        return (predictions > 0.5).astype(int).flatten()
