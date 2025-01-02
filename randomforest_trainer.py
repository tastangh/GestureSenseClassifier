# randomforest_trainer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os

class RandomForestTrainer:
    def __init__(self, input_dim, num_classes, n_estimators=100, max_depth=None, learning_rate=0.001, output_dir="./output"):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.label_binarizer = LabelBinarizer()
        
        self.model = Sequential()
        
        # Rastgele ağaçları simüle etmek için katmanlar oluştur
        for _ in range(n_estimators):
              if max_depth:
                for _ in range(max_depth):
                    self.model.add(Dense(input_dim, activation='relu')) # Her ağaç için input_dim boyutunda katman
              else :
                 self.model.add(Dense(input_dim, activation='relu')) # Ağaç sayısı kadar katman eklenir
        self.model.add(Dense(num_classes, activation='softmax')) # Son katman

        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = CategoricalCrossentropy()
        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32,
               early_stopping=False, patience=3):
         # One-hot encode labels
        y_train_encoded = self.label_binarizer.fit_transform(y_train)
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_binarizer.transform(y_val)
        else:
            y_val_encoded = None
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_encoded)).batch(batch_size)
        if X_val is not None and y_val is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_encoded)).batch(batch_size)
        else:
            val_dataset = None
            
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = []
            epoch_acc = []
            metric = tf.metrics.Accuracy() # Define metric
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss, y_pred = self.train_step(x_batch_train, y_batch_train)
                metric.update_state(tf.argmax(y_batch_train, axis=1), tf.argmax(y_pred, axis=1))
                acc = metric.result().numpy()
                epoch_loss.append(loss)
                epoch_acc.append(acc)
                print(f"  Batch {step + 1}/{len(train_dataset)}, Loss: {loss:.4f}, Accuracy: {acc:.4f}", end='\r')
            print() # Her batch'den sonra satır atla
            metric.reset_state() # reset the metric after each epoch

            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            print(f"  Eğitim Loss: {avg_loss:.4f}, Eğitim Accuracy: {avg_acc:.4f}")
            if val_dataset is not None:
              val_loss, val_acc = self.validate(val_dataset)
              history['val_loss'].append(val_loss)
              history['val_accuracy'].append(val_acc)
              print(f"  Doğrulama Loss: {val_loss:.4f}, Doğrulama Accuracy: {val_acc:.4f}")

        self.history = history
        print("\nEğitim Tamamlandı!")

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    def validate(self, val_dataset):
        val_loss_list = []
        val_acc_list = []
        metric = tf.metrics.Accuracy()
        for x_batch_val, y_batch_val in val_dataset:
          y_pred = self.model(x_batch_val, training=False)
          val_loss = self.loss_fn(y_batch_val, y_pred)
          metric.update_state(tf.argmax(y_batch_val, axis=1), tf.argmax(y_pred, axis=1))
          val_loss_list.append(val_loss)
          val_acc_list.append(metric.result().numpy())
        metric.reset_state()
        return np.mean(val_loss_list), np.mean(val_acc_list)
    
    def predict(self, X_test):
      predictions = self.model.predict(X_test)
      return np.argmax(predictions, axis=1)
      
    def plot_metrics(self):
      """
        Eğitim ve doğrulama kayıp/doğruluk metriklerini çizer.
        """
      if self.history is None:
        print("Henüz eğitim metrikleri mevcut değil!")
        return
      
      history = self.history
      # Kayıp grafiği
      plt.figure(figsize=(8, 6))
      plt.plot(history['loss'], label='Eğitim Kaybı')
      if 'val_loss' in history:
          plt.plot(history['val_loss'], label='Doğrulama Kaybı')
      plt.title("Eğitim ve Doğrulama Kaybı")
      plt.xlabel("Epoch")
      plt.ylabel("Kayıp")
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, "tf_randomforest_loss_plot.png"))
      plt.close()

      # Doğruluk grafiği
      plt.figure(figsize=(8, 6))
      plt.plot(history['accuracy'], label='Eğitim Doğruluk')
      if 'val_accuracy' in history:
          plt.plot(history['val_accuracy'], label='Doğrulama Doğruluk')
      plt.title("Eğitim ve Doğrulama Doğruluk")
      plt.xlabel("Epoch")
      plt.ylabel("Doğruluk")
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, "tf_randomforest_accuracy_plot.png"))
      plt.close()