import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from EMGDataset import EMGDataset
import torch
import torch.nn as nn
from lstm.LSTM_Model import LSTM_Model
import time
import torch.optim as optim
from utils import plot_metrics, plot_loss
import matplotlib.pyplot as plt

def read_data(file_path="/content/drive/MyDrive/ML_Final_Project/data/EMG-data.csv"):
    """
    Belirtilen dosya yolundan EMG verilerini CSV formatında okur.

    Parametreler:
    ------------
    file_path : str, optional
        CSV dosyasının yolu. Varsayılan değer "/content/drive/MyDrive/ML_Final_Project/data/EMG-data.csv" olarak ayarlanmıştır.

    Returns:
    --------
    pandas.DataFrame
        Okunan EMG verilerini içeren DataFrame.
    
    Örnek Kullanım:
    ---------------
    >>> data = read_data()
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Veri başarıyla okundu. Satır sayısı: {len(data)}")
        return data
    except FileNotFoundError:
        print(f"Hata: Dosya bulunamadı - {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Hata: Dosya boş - {file_path}")
        return None
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
        return None

def clean_data(data):
    """
    EMG verilerini temizler ve özellikler ile etiketleri ayırır.

    Bu fonksiyon aşağıdaki adımları gerçekleştirir:
    1. Veri setindeki eksik (NaN) değerleri kaldırır.
    2. Kanallardan (feature) veri seçer (sütun 1'den 8'e kadar olanlar).
    3. Sınıf etiketlerini (y) ayrıştırır.
    4. Temizlenen veri setinin uzunluğunu ekrana yazdırır.

    Parametreler:
    ------------
    data : pandas.DataFrame
        Ham EMG verilerini içeren DataFrame.

    Returns:
    --------
    tuple
        - X : pandas.DataFrame
            Özellikleri (kanal verilerini) içeren DataFrame.
        - y : pandas.Series
            Sınıf etiketlerini içeren Series.
    
    Örnek Kullanım:
    ---------------
    >>> X, y = clean_data(data)
    """
    if data is None or data.empty:
        print("Hata: Temizlenecek veri mevcut değil veya boş.")
        return None, None

    # Veri setindeki eksik değerleri kaldır
    initial_length = len(data)
    data.dropna(inplace=True)
    cleaned_length = len(data)
    print(f"Veri temizlendi. Kayıp veriler: {initial_length - cleaned_length}")

    # Kanallardan veri seç (sütun 1'den 8'e kadar olanlar)
    # 'time', 'class' ve 'label' sütunları hariç tutuluyor
    X = data.iloc[:, 1:9]
    y = data['class']  # Sınıf etiketleri

    # Temizlenen veri setinin uzunluklarını ekrana yazdır
    print(f"Özellik sayısı (X): {len(X)}")
    print(f"Etiket sayısı (y): {len(y)}")

    return X, y

def normalize_data(data):
    """
    Verileri standart ölçeklendirme yöntemiyle normalize eder.

    Standart ölçeklendirme, verilerin ortalamasını 0 ve standart sapmasını 1 yapar.
    Bu yöntem, verilerin farklı ölçeklerden kaynaklanan etkilerini azaltmak için kullanılır.

    Parametreler:
    ------------
    data : pandas.DataFrame veya numpy.ndarray
        Normalize edilecek ham veriler.

    Returns:
    --------
    numpy.ndarray
        Standart ölçeklendirilmiş veriler.
    
    Örnek Kullanım:
    ---------------
    >>> X_normalized = normalize_data(X)
    """
    if data is None or data.size == 0:
        print("Hata: Normalize edilecek veri mevcut değil veya boş.")
        return None

    # Standart ölçekleyici nesnesi oluştur
    standard_scaler = StandardScaler()

    try:
        # Verileri fit ve transform ederek normalize et
        normalized_data = standard_scaler.fit_transform(data)
        print("Veriler başarıyla normalize edildi.")
        return normalized_data
    except Exception as e:
        print(f"Veri normalize edilirken bir hata oluştu: {e}")
        return None
           
def split_data(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Veriyi eğitim, doğrulama ve test setlerine böler.

    Bu fonksiyon, verilen özellik (X) ve etiket (y) verilerini belirtilen oranlarda eğitim, doğrulama ve test setlerine böler.

    Parametreler:
    ------------
    X : pandas.DataFrame veya numpy.ndarray - Özellik verileri.
    
    y : pandas.Series veya numpy.ndarray - Etiket verileri.
    
    train_size : float, optional - Eğitim setinin toplam verinin hangi oranını alacağını belirler. Varsayılan değer 0.6'dır (yani %60).
    
    val_size : float, optional - Doğrulama setinin toplam verinin hangi oranını alacağını belirler. Varsayılan değer 0.2'dir (%20).
    
    test_size : float, optional - Test setinin toplam verinin hangi oranını alacağını belirler. Varsayılan değer 0.2'dir (%20).
    
    random_state : int, optional - Rastgele bölme işlemi için sabit bir tohum değeri. Böylece bölme işlemi tekrarlanabilir olur. Varsayılan değer 42'dir.
    
    Returns:
    --------
    tuple
        - X_train : pandas.DataFrame veya numpy.ndarray
            Eğitim seti için özellik verileri.
        - X_val : pandas.DataFrame veya numpy.ndarray
            Doğrulama seti için özellik verileri.
        - X_test : pandas.DataFrame veya numpy.ndarray
            Test seti için özellik verileri.
        - y_train : pandas.Series veya numpy.ndarray
            Eğitim seti için etiket verileri.
        - y_val : pandas.Series veya numpy.ndarray
            Doğrulama seti için etiket verileri.
        - y_test : pandas.Series veya numpy.ndarray
            Test seti için etiket verileri.
    
    Örnek Kullanım:
    ---------------
    >>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    """
    # İlk olarak, veriyi eğitim ve geçici setlere (doğrulama + test) bölüyoruz
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_size + test_size), 
        random_state=random_state,
        stratify=y  # Sınıf dağılımının korunmasını sağlar
    )
    
    val_ratio = val_size / (val_size + test_size)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1 - val_ratio), 
        random_state=random_state,
        stratify=y_temp  # Sınıf dağılımının korunmasını sağlar
    )
    
    assert len(X_train) + len(X_val) + len(X_test) == len(X), "Veri bölme hatası: Toplam veri sayısı eşleşmiyor."
    
    print(f"Veri başarıyla bölündü:")
    print(f"Eğitim seti: {len(X_train)} örnek")
    print(f"Doğrulama seti: {len(X_val)} örnek")
    print(f"Test seti: {len(X_test)} örnek")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Eğitim, doğrulama ve test setlerinden PyTorch veri kümeleri oluşturur.

    Bu fonksiyon, verilen eğitim, doğrulama ve test verilerini kullanarak EMGDataset sınıfını kullanarak veri kümeleri oluşturur.

    Parametreler:
    ------------
    X_train : pandas.DataFrame veya numpy.ndarray
        Eğitim seti için özellik verileri.
    
    y_train : pandas.Series veya numpy.ndarray
        Eğitim seti için etiket verileri.
    
    X_val : pandas.DataFrame veya numpy.ndarray
        Doğrulama seti için özellik verileri.
    
    y_val : pandas.Series veya numpy.ndarray
        Doğrulama seti için etiket verileri.
    
    X_test : pandas.DataFrame veya numpy.ndarray
        Test seti için özellik verileri.
    
    y_test : pandas.Series veya numpy.ndarray
        Test seti için etiket verileri.
    
    Returns:
    --------
    tuple
        - train_dataset : EMGDataset
            Eğitim seti için oluşturulan veri kümesi.
        - val_dataset : EMGDataset
            Doğrulama seti için oluşturulan veri kümesi.
        - test_dataset : EMGDataset
            Test seti için oluşturulan veri kümesi.
    
    Örnek Kullanım:
    ---------------
    >>> train_dataset, val_dataset, test_dataset = create_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Eğitim veri kümesini oluştur
    train_dataset = EMGDataset(X_train, y_train)
    print(f"Eğitim veri kümesi oluşturuldu: {len(train_dataset)} örnek")
    
    # Doğrulama veri kümesini oluştur
    val_dataset = EMGDataset(X_val, y_val)
    print(f"Doğrulama veri kümesi oluşturuldu: {len(val_dataset)} örnek")
    
    # Test veri kümesini oluştur
    test_dataset = EMGDataset(X_test, y_test)
    print(f"Test veri kümesi oluşturuldu: {len(test_dataset)} örnek")
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    """
    Eğitim, doğrulama ve test veri kümelerinden PyTorch DataLoader nesneleri oluşturur.

    Bu fonksiyon, verilen veri kümelerini kullanarak PyTorch DataLoader nesneleri oluşturur. DataLoader'lar, model eğitimi ve değerlendirmesi sırasında verilerin verimli bir şekilde işlenmesini sağlar.

    Parametreler:
    ------------
    train_dataset : EMGDataset
        Eğitim veri kümesi.
    
    val_dataset : EMGDataset
        Doğrulama veri kümesi.
    
    test_dataset : EMGDataset
        Test veri kümesi.
    
    batch_size : int, optional
        Her bir batch'teki örnek sayısı. Varsayılan değer 64'tür.
    
    Returns:
    --------
    tuple
        - train_loader : DataLoader
            Eğitim veri kümesi için DataLoader.
        - val_loader : DataLoader
            Doğrulama veri kümesi için DataLoader.
        - test_loader : DataLoader
            Test veri kümesi için DataLoader.
    
    Örnek Kullanım:
    ---------------
    >>> train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)
    """
    # Eğitim DataLoader'ını oluştur
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True  # Eğitim verilerini karıştırmak için shuffle=True
    )
    print(f"Eğitim DataLoader oluşturuldu: {len(train_loader)} batch")
    
    # Doğrulama DataLoader'ını oluştur
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False  # Doğrulama verilerini karıştırmaya gerek yok
    )
    print(f"Doğrulama DataLoader oluşturuldu: {len(val_loader)} batch")
    
    # Test DataLoader'ını oluştur
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False  # Test verilerini karıştırmaya gerek yok
    )
    print(f"Test DataLoader oluşturuldu: {len(test_loader)} batch")
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_accuracy = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        start_train_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        end_train_time = time.time()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        training_times.append(end_train_time - start_train_time)

        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        start_test_time = time.time()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        end_test_time = time.time()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        testing_times.append(end_test_time - start_test_time)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}')

        # Save the model with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            
if __name__ == "__main__":
    
    # Data operations
    
    data = read_data("data.csv")
    X, y = clean_data(data)
    X = normalize_data(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    train_dataset, val_dataset, test_dataset = create_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Hyperparameters
    hidden_size = 128
    num_layers = 3
    dropout_rate = 0.4
    learning_rate = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device - ", device)

    # Create the model, define the loss function, and optimizer
    model = LSTM_Model(input_size=1, hidden_size=hidden_size, output_size=len(y.unique()), num_layers=num_layers,dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    training_times = []
    testing_times = []
    num_epochs = 5

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
    plot_loss(train_losses, val_losses, num_epochs)
    plot_metrics(val_accuracies, val_f1_scores, num_epochs)
    model.load_state_dict(torch.load('best_model.pth'))
