import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Dosya yolları
RAW_DATA_PATH = "dataset/emg_data.csv"
FEATURES_DATA_PATH = "features_emg_data.csv"
RESULTS_DIR = "results/model_comparisons_dynamic_lr"

os.makedirs(RESULTS_DIR, exist_ok=True)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=100):
    """
    Modeli eğitir ve doğrulama seti üzerinde değerlendirir.
    """
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Doğrulama setinde değerlendirme
        model.eval()
        val_accuracy = 0.0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_accuracy += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy /= val_total
        val_loss /= val_total

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        scheduler.step(val_loss)  # ReduceLROnPlateau kullanımı

    return model

def prepare_data(X, y, batch_size=32):
    """
    Veriyi PyTorch DataLoader formatına dönüştürür.
    """
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def evaluate_scenario(data_path, label_column="Gesture_Class", test_size=0.2, num_epochs=100):
    """
    Belirli bir veri seti ile modeli eğitir ve değerlendirir.
    """
    logger.info(f"{data_path} yükleniyor...")
    data = pd.read_csv(data_path)
    X = data.drop(columns=[label_column]).values
    y = data[label_column].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalize veriler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # PyTorch DataLoader
    train_loader = prepare_data(X_train, y_train)
    val_loader = prepare_data(X_val, y_val)

    # Model oluşturma
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = LogisticRegressionModel(input_dim, num_classes).to(device)

    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Modeli eğitme
    logger.info("Model eğitiliyor...")
    trained_model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=num_epochs)

    # Test setinde değerlendirme
    logger.info("Test seti değerlendirmesi yapılıyor...")
    val_loader = prepare_data(X_val, y_val)
    trained_model.eval()
    test_accuracy = 0.0
    test_total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_accuracy += (preds == labels).sum().item()
            test_total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_accuracy /= test_total
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Classification Report:\n{classification_report(y_true, y_pred)}")
    return test_accuracy

def main():
    logger.info("\n--- Ham Veri (emg_data.csv) Model Değerlendirmesi ---")
    raw_accuracy = evaluate_scenario(RAW_DATA_PATH, num_epochs=100)

    logger.info("\n--- Özellik Çıkarımı (features_emg_data.csv) Model Değerlendirmesi ---")
    features_accuracy = evaluate_scenario(FEATURES_DATA_PATH, num_epochs=100)

    logger.info("\n--- SONUÇLAR KARŞILAŞTIRMASI ---")
    logger.info(f"Ham Veri Accuracy: {raw_accuracy}")
    logger.info(f"Özellik Çıkarımı Accuracy: {features_accuracy}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
