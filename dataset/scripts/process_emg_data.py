import pandas as pd

# Dosyanın adını belirtin
file_path = "emg_data.csv"  # Dosya yolunu kendi dosyanıza göre güncelleyin

# CSV dosyasını oku
df = pd.read_csv(file_path, header=None)

# Sütun isimlerini oluştur
sensor_columns = [f"S{sensor_num}_R{reading_num}" 
                  for reading_num in range(1, 9) 
                  for sensor_num in range(1, 9)]

# Sütun isimlerini atayın (Son sütun 'Gesture_Class')
df.columns = sensor_columns + ["Gesture_Class"]

# İlk birkaç satırı kontrol edin
print(df.head())

# Dosyanın işlenmiş halini kaydetmek isterseniz:
df.to_csv("labeled_emg_data.csv", index=False)
print("Dosya başarıyla 'labeled_emg_data.csv' olarak kaydedildi.")
