from base_trainer import BaseTrainer
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy

class LSTMTrainer(BaseTrainer):
    def __init__(self, input_shape, num_classes, lstm_units=50, dropout_rate=0.2, learning_rate=0.001, output_dir="./output"):
        super().__init__(input_shape[1], num_classes, output_dir)
        self.model = Sequential([
            LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        self.loss_fn = CategoricalCrossentropy()