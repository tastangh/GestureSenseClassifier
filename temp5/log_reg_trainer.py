from base_trainer import BaseTrainer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
class LogRegTrainer(BaseTrainer):
    def __init__(self, input_dim, num_classes=8, output_dir="./output"):
        super().__init__(input_dim, num_classes, output_dir)
        self.model = Sequential([
            Dense(num_classes, activation='softmax', input_dim=input_dim)
        ])
        self.loss_fn = CategoricalCrossentropy()