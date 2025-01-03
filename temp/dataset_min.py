import pandas as pd
import itertools

# 1. Veriyi Yükleme
data = pd.read_csv("dataset/EMG-data.csv")

# 2. "class = 0" Satırlarını Kaldırma
data_cleaned = data[data["class"] != 0]

# 3. Örneklerin Çıkartılması
# Benzersiz label ve class değerlerini al
labels = data_cleaned["label"].unique()
classes = data_cleaned["class"].unique()

# Tüm sınıf çiftlerini oluştur
class_combinations = list(itertools.combinations(classes, 2))

# Örneklerin tutulacağı liste
samples = []

# Her label için işlemleri gerçekleştir
for label in labels:
    label_data = data_cleaned[data_cleaned["label"] == label]  # Her bir label için veri
    for class_pair in class_combinations:
        class_1_data = label_data[label_data["class"] == class_pair[0]]
        class_2_data = label_data[label_data["class"] == class_pair[1]]
        
        # Her sınıf çiftinden en az bir örnek al
        if not class_1_data.empty and not class_2_data.empty:
            samples.append(class_1_data.sample(1, random_state=42))
            samples.append(class_2_data.sample(1, random_state=42))

# Tüm örnekleri birleştir
result_data = pd.concat(samples)

# 4. Çıktıyı Kaydetme
result_data.to_csv("sampled_emg_data.csv", index=False)
print(f"Oluşturulan örnek sayısı: {result_data.shape[0]}")
