from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

class LSTMModel:
    def __init__(self, n_steps, n_features, saved_model=False):
        """
        LSTM modelini başlatır.
        
        :param n_steps: Zaman adımlarının sayısı (örneğin 8 zaman adımı).
        :param n_features: Özellik sayısı (örneğin 8 sensör).
        :param saved_model: Eğer daha önce eğitilmiş bir model yüklenecekse True.
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.saved_model = saved_model
        self.model = None

        # Eğer daha önce eğitilmiş model yüklenecekse
        if self.saved_model:
            print("Trained LSTM Model is loading...\n")
            self.model = load_model("results/lstm_model.h5")
        else:
            print("LSTM Training Session has begun...\n")
            self.model = self._build_model()

    def _build_model(self):
        """LSTM modelini oluşturur."""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(0.2))

        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(50))
        model.add(Dropout(0.2))

        model.add(Dense(64))
        model.add(Dense(128))

        model.add(Dense(4, activation="softmax"))
        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Modeli eğitir ve kaydeder."""
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

        # Eğitim kaybını görselleştir
        plt.plot(history.history['loss'])
        plt.title('Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Epok')
        plt.legend(['Eğitim'], loc='upper left')
        plt.show()

        # Eğitilmiş modeli kaydet
        self.model.save("results/lstm_model.h5")
        print("Model saved to disk.\n")

    def predict(self, X_test):
        """Test seti üzerinde tahmin yapar."""
        y_pred = self.model.predict(X_test)
        return np.argmax(y_pred, axis=1)
