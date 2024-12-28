import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Veriyi Yükleme
data = pd.read_csv("dataset/EMG-data.csv")

# 2. "class = 0" Satırlarını Kaldırma
data_cleaned = data[data["class"] != 0]

# 3. Özelliklerin ve Hedef Değişkenin Seçilmesi
X = data_cleaned[["channel1", "channel2", "channel3", "channel4", 
                  "channel5", "channel6", "channel7", "channel8"]]
y = data_cleaned["class"]

# 4. Veriyi Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Logistic Regression Modeli
model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 6. Tahmin Yapma ve Performans Ölçümü
y_pred = model.predict(X_test)

# Doğruluk
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy * 100:.2f}%")

# Ayrıntılı Sınıflandırma Raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in sorted(y.unique())]))
