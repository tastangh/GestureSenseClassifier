from base_trainer import BaseTrainer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
class DecisionTreeTrainer(BaseTrainer):
     def __init__(self, input_dim, num_classes, max_depth=None, learning_rate=0.001, output_dir="./output"):
        super().__init__(input_dim, num_classes, output_dir)
        self.max_depth = max_depth
        self.model = Sequential()
        
        # Katmanları dinamik oluştur
        for _ in range(max_depth if max_depth else 5):  # Belirli bir derinlik veya varsayılan derinlik (örneğin, 5)
            self.model.add(Dense(input_dim, activation='relu')) # Her katmanda input_dim boyutunu al.
        self.model.add(Dense(num_classes, activation='softmax'))
        self.loss_fn = CategoricalCrossentropy()