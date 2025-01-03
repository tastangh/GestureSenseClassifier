# base_trainer.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class BaseTrainer:
    """
    Base class for all model trainers.
    """
    def __init__(self, input_dim, num_classes, output_dir):
        """
        Initializes the base trainer.
        :param input_dim: Input dimension.
        :param num_classes: Number of classes.
        :param output_dir: Directory to save output files.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.history = None
        self.label_binarizer = LabelBinarizer()

    def _create_optimizer(self, optimizer_type, learning_rate):
        if optimizer_type == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
             return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32,
               optimizer_type='adam', learning_rate=0.001,
              early_stopping=False, patience=3,
              learning_rate_scheduling=False, factor=0.1, min_lr=1e-6):
        """
        Trains the model.
        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_val: Validation data (optional).
        :param y_val: Validation labels (optional).
        :param epochs: Number of training epochs.
        :param batch_size: Batch size.
        :param optimizer_type: Optimization algorithm ('adam' or 'sgd')
        :param learning_rate: Learning rate.
        :param early_stopping: Enable early stopping.
        :param patience: Early stopping patience.
        :param learning_rate_scheduling: Enable learning rate scheduling.
        :param factor: Learning rate reduction factor.
        :param min_lr: Minimum learning rate.
        """
         # One-hot encode labels
        y_train_encoded = self.label_binarizer.fit_transform(y_train)
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_binarizer.transform(y_val)
        else:
            y_val_encoded = None
        
        self.optimizer = self._create_optimizer(optimizer_type, learning_rate)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_encoded)).batch(batch_size)
        if X_val is not None and y_val is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_encoded)).batch(batch_size)
        else:
            val_dataset = None
            
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Callbacks
        callbacks = self._create_callbacks(early_stopping, patience, learning_rate_scheduling, factor, min_lr)

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

    def _create_callbacks(self, early_stopping, patience, learning_rate_scheduling, factor, min_lr):
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
        if learning_rate_scheduling:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience // 2, min_lr=min_lr))
        return callbacks

    @tf.function
    def train_step(self, x, y):
        """
        Performs one training step on a batch of data.
        :param x: Input batch.
        :param y: True labels for batch.
        :return: Loss value and predicted batch.
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, y_pred

    def validate(self, val_dataset):
        """
        Validates the model.
        :param val_dataset: Validation data.
        :return: Validation loss and accuracy.
        """
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
        """
        Predicts on the test set.
        :param X_test: Test data.
        :return: Predicted labels.
        """
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def plot_metrics(self, model_name, model_params):
            """
            Plots training and validation metrics.
            :param model_name: Name of the model.
            :param model_params: Model parameters
            """
            if self.history is None:
                print("Henüz eğitim metrikleri mevcut değil!")
                return

            history = self.history
            
            # Parametreleri dosya adına dahil et
            params_str = "_".join([f"{key}-{value}" for key, value in model_params.items()])

            # Kayıp grafiği
            plt.figure(figsize=(8, 6))
            plt.plot(history['loss'], label='Eğitim Kaybı')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Doğrulama Kaybı')
            plt.title("Eğitim ve Doğrulama Kaybı")
            plt.xlabel("Epoch")
            plt.ylabel("Kayıp")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_loss_plot_{params_str}.png"))
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
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_accuracy_plot_{params_str}.png"))
            plt.close()