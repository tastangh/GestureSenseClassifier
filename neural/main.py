import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

from EMGDataset import EMGDataset
from NeuralNetwork import SimpleNN
from utils import plot_loss, plot_confusion_matrix, plot_metrics
from train import train_model, read_data, clean_data, normalize_data, split_data, create_datasets, create_data_loaders
from eval import evaluate_model


file_path = "data.csv"
data = read_data(file_path=file_path)
X, y = clean_data(data)
X = normalize_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
train_dataset, val_dataset, test_dataset = create_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

# Hyperparametreler
hidden_sizes = [256, 128, 64]
dropout_rate = 0.5
learning_rate = 0.01
num_epochs = 20
# **Sınıf sayısını güncelleme**
num_classes = len(y.unique())  # Örneğin, 7 sınıf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Kayıp Fonksiyonu ve Optimizatör
model = SimpleNN(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=num_classes, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Modeli Eğitme
train_losses, val_losses, val_accuracies, val_f1_scores, training_times, testing_times = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

plot_loss(train_losses, val_losses, num_epochs)
plot_metrics(val_accuracies, val_f1_scores, num_epochs)

model.load_state_dict(torch.load('best_simple_nn.pth'))

y_true, y_pred, test_time = evaluate_model(model, test_loader)

class_names = sorted(np.unique(y))
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=False,
                      title='Confusion Matrix for Test Set')
