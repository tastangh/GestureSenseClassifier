import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

class LSTMTrainer:
    """
    LSTM modeli eğitmek ve metrik grafikleri çizmek için bir sınıf.
    """
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        LSTMTrainer sınıfını başlatır.
        :param input_shape: Giriş verisinin şekli (örneğin, (zaman_adımı, özellik_sayısı))
        :param lstm_units: LSTM hücre sayısı
        :param dropout_rate: Dropout oranı
        :param learning_rate: Öğrenme oranı
        """
        self.model = Sequential([
            LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
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

    def plot_metrics(self):
        """
        Eğitim ve doğrulama kayıp/doğruluk metriklerini çizer.
        """
        if self.history is None:
            print("Henüz eğitim metrikleri mevcut değil!")
            return

        history = self.history.history

        # Kayıp grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['loss'], label='Eğitim Kaybı')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Doğrulama Kaybı')
        plt.title("Eğitim ve Doğrulama Kaybı")
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp")
        plt.legend()
        plt.show()

        # Doğruluk grafiği
        plt.figure(figsize=(8, 6))
        plt.plot(history['accuracy'], label='Eğitim Doğruluk')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Doğrulama Doğruluk')
        plt.title("Eğitim ve Doğrulama Doğruluk")
        plt.xlabel("Epoch")
        plt.ylabel("Doğruluk")
        plt.legend()
        plt.show()

# Örnek Kullanım
# input_shape = (100, 10)  # Zaman adımı: 100, Özellik sayısı: 10
# trainer = LSTMTrainer(input_shape=input_shape, lstm_units=64)
# trainer.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=20)
# trainer.plot_metrics()
