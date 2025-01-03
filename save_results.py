# save_results.py
import pandas as pd

def save_results_to_excel(output_dir, model_name, model_params, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Model sonuçlarını bir Excel dosyasına kaydeder.
    :param output_dir: Çıktı dosyalarının kaydedileceği klasör
    :param model_name: Modelin adı (örneğin, 'LogisticRegression')
    :param model_params: Modelin parametreleri (sözlük formatında)
    :param train_loss: Eğitim kaybı
    :param train_accuracy: Eğitim doğruluğu
    :param val_loss: Doğrulama kaybı
    :param val_accuracy: Doğrulama doğruluğu
    """
    file_path = f"{output_dir}/model_results.xlsx"
    
    # Yeni sonuçlar için bir veri çerçevesi oluştur
    new_entry = {
        "Model Name": model_name,
        "Model Parameters": str(model_params),
        "Training Loss": train_loss,
        "Training Accuracy": train_accuracy,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy
    }
    new_df = pd.DataFrame([new_entry])
    
    # Eğer dosya zaten varsa, eski verileri yükle ve yeni veriyi ekle
    try:
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = new_df
    
    # Sonuçları Excel dosyasına kaydet
    updated_df.to_excel(file_path, index=False)
    print(f"Sonuçlar '{file_path}' dosyasına kaydedildi.")

def save_results_to_csv(output_dir, model_name, model_params, train_accuracy):
    """
    Model sonuçlarını bir CSV dosyasına kaydeder.
    :param output_dir: Çıktı dosyalarının kaydedileceği klasör
    :param model_name: Modelin adı (örneğin, 'LogisticRegression')
    :param model_params: Modelin parametreleri (sözlük formatında)
    :param train_accuracy: Eğitim doğruluğu
    """
    file_path = f"{output_dir}/model_results.csv"
    
    # Yeni sonuçlar için bir veri çerçevesi oluştur
    new_entry = {
        "Model Name": model_name,
        "Model Parameters": str(model_params),
        "Training Accuracy": train_accuracy
    }
    new_df = pd.DataFrame([new_entry])
    
    # Eğer dosya zaten varsa, eski verileri yükle ve yeni veriyi ekle
    try:
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = new_df
    
    # Sonuçları CSV dosyasına kaydet
    updated_df.to_csv(file_path, index=False)
    print(f"Sonuçlar '{file_path}' dosyasına kaydedildi.")

def save_results_to_txt(output_dir, model_name, model_params, train_accuracy):
    """
    Model sonuçlarını bir TXT dosyasına kaydeder.
    :param output_dir: Çıktı dosyalarının kaydedileceği klasör
    :param model_name: Modelin adı (örneğin, 'LogisticRegression')
    :param model_params: Modelin parametreleri (sözlük formatında)
    :param train_accuracy: Eğitim doğruluğu
    """
    file_path = f"{output_dir}/model_results.txt"
    with open(file_path, "a") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Parameters: {model_params}\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write("-" * 50 + "\n")
    print(f"Sonuçlar '{file_path}' dosyasına kaydedildi.")