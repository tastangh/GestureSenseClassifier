import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import time
from tqdm import tqdm  # İlerleme çubuğu için

# Veri setini yükle
data = pd.read_csv('dataset/EMG-data.csv')

# Kanal verilerini seç
X = data.iloc[:, 1:9]  # channel1'den channel8'e kadar olan sütunlar
y = data['class']  # Sınıf etiketleri

# Veriyi normalize et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veriyi eğitim, doğrulama ve test setlerine ayır
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Veriyi PyTorch dataset formatına dönüştür
class EMGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(-1)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Dataset ve DataLoader oluştur
train_dataset = EMGDataset(X_train, y_train)
val_dataset = EMGDataset(X_val, y_val)
test_dataset = EMGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN-LSTM modelini tanımla
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Hiperparametreler
hidden_size = 128
num_layers = 3
dropout_rate = 0.4
learning_rate = 0.0005

# Cihazı belirle (CPU veya GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA kullanılıp kullanılmadığını loglama
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Model, loss fonksiyonu ve optimizer tanımla
model = CNN_LSTM_Model(input_size=1, hidden_size=hidden_size, output_size=len(y.unique()), num_layers=num_layers, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Eğitim ve doğrulama
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        # Eğitim Döngüsü
        for i, (inputs, labels) in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"  Training Loss: {epoch_loss:.4f}")

        # Doğrulama Döngüsü
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        val_progress = tqdm(val_loader, desc="Validating")

        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        avg_val_loss = val_loss / len(val_loader)
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # En iyi modeli kaydet
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Best model saved.")

# Test modeli
def evaluate_model(model, test_loader):
    print("Testing Model...")
    model.eval()
    all_labels = []
    all_preds = []
    test_progress = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, labels in test_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Modeli eğit ve değerlendir
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
evaluate_model(model, test_loader)
