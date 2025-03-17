# Gerekli Kütüphanelerin İçe Aktarılması
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Veri Yükleme
# Dosya yolunu kendi verinizin bulunduğu konuma göre ayarlayın
df = pd.read_csv(DATASET_FILE_PATH)
df = df.drop('label', axis=1)
print(f"Veri Setindeki Toplam Satır Sayısı: {len(df)}")

# Verinin İlk Hali
print("\nVeri Setinin İlk 5 Satırı:")
print(df.head())

# Hedef ve Özelliklerin Belirlenmesi
# 'class' sütununu kendi veri setinizdeki hedef sütunun adıyla değiştirin
y = df['class']
X = df.drop('class', axis=1)

# Eğitim, Doğrulama ve Test Setlerine Ayırma (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nEğitim, Doğrulama ve Test Setlerinin Boyutları:")
print(f"Eğitim Seti: {X_train.shape}, {y_train.shape}")
print(f"Doğrulama Seti: {X_val.shape}, {y_val.shape}")
print(f"Test Seti: {X_test.shape}, {y_test.shape}")

# Özellik Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Confusion Matrix Çizme Fonksiyonu
def plot_confusion_matrix(y_true, y_pred, classes, title):
    """
    Confusion Matrix'i çizmek için bir fonksiyon.

    Parameters:
    - y_true: Gerçek etiketler
    - y_pred: Tahmin edilen etiketler
    - classes: Sınıf isimleri
    - title: Grafik başlığı
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.show()

# Performans Değerlendirmesi Fonksiyonu
def evaluate_model(name, y_true, y_pred, classes):
    """
    Model performansını değerlendirmek ve Confusion Matrix'i çizmek için bir fonksiyon.

    Parameters:
    - name: Model adı
    - y_true: Gerçek etiketler
    - y_pred: Tahmin edilen etiketler
    - classes: Sınıf isimleri
    """
    print(f"\n{name} Performans Değerlendirmesi:")
    print("Doğruluk Skoru:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, classes, f"{name} Confusion Matrix")

# Sınıf İsimlerini Belirleme (Eğer Sınıf Etiketleriniz Sayısal Değilse)
# Örneğin, eğer sınıflar 0-6 arası sayısal ise, bunları kullanabilirsiniz
classes = sorted(y.unique())

# Bagging Yöntemi ile Karar Ağacı
print("\nBagging Classifier ile Karar Ağacı Eğitiliyor...")
base_clf = DecisionTreeClassifier(random_state=42)
bagging_clf = BaggingClassifier(
    estimator=base_clf,  # 'base_estimator' yerine 'estimator' kullanıldı
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_val)
bagging_accuracy = accuracy_score(y_val, y_pred_bagging)
print(f"Bagging Classifier Doğruluk Skoru (Doğrulama Seti): {bagging_accuracy:.4f}")

# Random Forest ile Bagging
print("\nRandom Forest Classifier Eğitiliyor...")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_val)
rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Doğruluk Skoru (Doğrulama Seti): {rf_accuracy:.4f}")

# AdaBoost Yöntemi
print("\nAdaBoost Classifier Eğitiliyor...")
ada_clf = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_val)
ada_accuracy = accuracy_score(y_val, y_pred_ada)
print(f"AdaBoost Classifier Doğruluk Skoru (Doğrulama Seti): {ada_accuracy:.4f}")

# Gradient Boosting Yöntemi
print("\nGradient Boosting Classifier Eğitiliyor...")
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_val)
gb_accuracy = accuracy_score(y_val, y_pred_gb)
print(f"Gradient Boosting Classifier Doğruluk Skoru (Doğrulama Seti): {gb_accuracy:.4f}")

# XGBoost Yöntemi (İsteğe Bağlı)
print("\nXGBoost Classifier Eğitiliyor...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
print(f"XGBoost Classifier Doğruluk Skoru (Doğrulama Seti): {xgb_accuracy:.4f}")

# Modellerin Performansını Doğrulama Seti Üzerinde Değerlendirme
evaluate_model("Bagging Classifier", y_val, y_pred_bagging, classes)
evaluate_model("Random Forest", y_val, y_pred_rf, classes)
evaluate_model("AdaBoost Classifier", y_val, y_pred_ada, classes)
evaluate_model("Gradient Boosting Classifier", y_val, y_pred_gb, classes)
evaluate_model("XGBoost Classifier", y_val, y_pred_xgb, classes)  # XGBoost için

# Doğruluk Skorlarının Karşılaştırılması
accuracy_scores = {
    'Bagging': bagging_accuracy,
    'Random Forest': rf_accuracy,
    'AdaBoost': ada_accuracy,
    'Gradient Boosting': gb_accuracy,
    'XGBoost': xgb_accuracy  # XGBoost için
}

# Skorları DataFrame'e Çevirme
acc_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=acc_df, palette='viridis')
plt.title('Modellerin Doğruluk Skorları Karşılaştırması (Doğrulama Seti)')
plt.xlim(0, 1)
for index, value in enumerate(acc_df['Accuracy']):
    plt.text(value + 0.01, index, f"{value:.4f}")
plt.xlabel('Doğruluk Skoru')
plt.ylabel('Model')
plt.show()

# Çapraz Doğrulama (İsteğe Bağlı)
print("\nRandom Forest için 5 Katlı Çapraz Doğrulama (Eğitim Seti):")
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Ortalama CV Doğruluk Skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Hiperparametre Ayarlaması (Örnek: Random Forest için GridSearchCV)
print("\nRandom Forest için Hiperparametre Ayarlaması Yapılıyor...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi CV Doğruluk Skoru: {grid_search.best_score_:.4f}")

# En İyi Model ile Test Seti Değerlendirmesi
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("\nEn İyi Random Forest Modelinin Test Seti Performansı:")
print(classification_report(y_test, y_pred_best_rf))

# Test Seti Üzerinde Genel Performans
evaluate_model("En İyi Random Forest (Test Seti)", y_test, y_pred_best_rf, classes)
