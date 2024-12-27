import pandas as pd

# Dosya yolları
file_paths = [
    "../0.csv",  # İlk dosya
    "../1.csv",  # İkinci dosya
    "../2.csv",  # Üçüncü dosya
    "../3.csv",  # Dördüncü dosya
]

# Tüm veriyi saklama listesi
datasets = []

# Dosyaları okuyup sütunları etiketleme
for file_path in file_paths:
    df = pd.read_csv(file_path, header=None)
    sensor_columns = [f"S{sensor_num}_R{reading_num}" 
                      for reading_num in range(1, 9) 
                      for sensor_num in range(1, 9)]
    df.columns = sensor_columns + ["Gesture_Class"]
    datasets.append(df)

# Dosyaları birleştirme
combined_df = pd.concat(datasets, ignore_index=True)

# Veriyi karıştırma
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Karıştırılmış veriyi kaydetme
output_file = "../emg_data.csv"
shuffled_df.to_csv(output_file, index=False)

# Çıkış bilgisi
print(f"Karıştırılmış veri '{output_file}' dosyasına kaydedildi.")
print(f"Toplam satır sayısı: {len(shuffled_df)}")
print("Karıştırılmış verinin ilk 5 satırı:")
print(shuffled_df.head())
