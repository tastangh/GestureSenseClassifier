from base_trainer import BaseTrainer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy

class RandomForestTrainer(BaseTrainer):
    def __init__(self, input_dim, num_classes, n_estimators=100, max_depth=None, learning_rate=0.001, output_dir="./output"):
        super().__init__(input_dim, num_classes, output_dir)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.model = Sequential()
        for _ in range(n_estimators):
            if max_depth:
                for _ in range(max_depth):
                    self.model.add(Dense(input_dim, activation='relu'))
            else:
                self.model.add(Dense(input_dim, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.loss_fn = CategoricalCrossentropy()