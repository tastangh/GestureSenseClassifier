from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os

class LSTMModel:
    def __init__(self, n_steps, n_features, saved_model=False, save_path="results"):
        """
        LSTM modelini başlatır.
        
        :param n_steps: Zaman adımlarının sayısı (örneğin 8 zaman adımı).
        :param n_features: Özellik sayısı (örneğin 8 sensör).
        :param saved_model: Eğer daha önce eğitilmiş bir model yüklenecekse True.
        :param save_path: Grafik ve modelin kaydedileceği dizin.
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.saved_model = saved_model
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.model = None

        # Eğer daha önce eğitilmiş model yüklenecekse
        if self.saved_model:
            print("Trained LSTM Model is loading...\n")
            self.model = load_model(os.path.join(self.save_path, "lstm_model.h5"))
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

        # Eğitim kaybı grafiğini çiz ve kaydet
        plt.plot(history.history['loss'])
        plt.title('Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Epok')
        plt.legend(['Eğitim'], loc='upper left')

        # Kaydetme işlemi
        loss_plot_path = os.path.join(self.save_path, "lstm_loss_plot.png")
        plt.savefig(loss_plot_path)
        print(f"Loss grafiği kaydedildi: {loss_plot_path}")

        plt.show()
        plt.close()

        # Eğitilmiş modeli kaydet
        model_save_path = os.path.join(self.save_path, "lstm_model.h5")
        self.model.save(model_save_path)
        print(f"model {model_save_path}'a kayededildi.\n")

    def predict(self, X_test):
        """Test seti üzerinde tahmin yapar."""
        y_pred = self.model.predict(X_test)
        return np.argmax(y_pred, axis=1)
