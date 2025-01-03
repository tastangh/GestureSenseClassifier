from base_trainer import BaseTrainer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
class ANNTrainer(BaseTrainer):
    def __init__(self, input_dim, num_classes, hidden_layers, dropout_rate=0.2, learning_rate=0.001, output_dir="./output"):
        super().__init__(input_dim, num_classes, output_dir)
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.loss_fn = CategoricalCrossentropy()