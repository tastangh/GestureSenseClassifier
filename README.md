# GestureSense README

Bu proje, makine öğrenimi ve derin öğrenme algoritmalarını kullanarak model eğitimi ve değerlendirmesi yapmayı amaçlamaktadır. Proje kapsamında dört farklı makine öğrenimi algoritması (Karar Ağacı, Lojistik Regresyon, Random Forest ve SVM) ile iki farklı derin öğrenme yaklaşımı (LSTM ve genel sinir ağları) kullanılmıştır.

## Proje Yapısı

Proje aşağıdaki gibi bir klasör yapısına sahiptir:
├── decision_tree_trainer.py
├── log_reg_trainer.py
├── random_forest_trainer.py
├── svm_trainer.py
├── lstm/
│  ── EMGDataset.py
│ ├── eval.py
│ ├── LSTM_Model.py
│ ├── main.py
│ ├── train.py
│ └── utils.py
├── neural/
│ ├── EMGDataset.py
│ ├── eval.py
│ ├── main.py
│ ├── NeuralNetwork.py
│ ├── train.py
│ └── utils.py
├── results/
│ └── # Modeller, metrikler ve görseller buraya kaydedilir
├── requirements.txt
└── README.md

*   **`boosting_code.py`**: Boosting algoritmaları (AdaBoost, Gradient Boosting, XGBoost) ve Bagging yöntemiyle model eğitimi için kullanılan Python dosyası.
*   **`decision_tree_trainer.py`**: Karar Ağacı algoritması ile model eğitimi için kullanılan Python dosyası.
*   **`log_reg_trainer.py`**: Lojistik Regresyon algoritması ile model eğitimi için kullanılan Python dosyası.
*   **`random_forest_trainer.py`**: Random Forest algoritması ile model eğitimi için kullanılan Python dosyası.
*   **`svm_trainer.py`**: SVM algoritması ile model eğitimi için kullanılan Python dosyası.
*   **`lstm/`**: LSTM tabanlı derin öğrenme modelleri için dosyaları içerir.
    *   **`EMGDataset.py`**: EMG veri kümesi yükleme ve ön işleme işlemleri için kullanılır.
    *   **`eval.py`**: LSTM modelini değerlendirmek için kullanılan dosya.
    *   **`LSTM_Model.py`**: LSTM model mimarisini tanımlar.
    *   **`main.py`**: LSTM eğitim sürecini başlatır.
    *   **`train.py`**: LSTM modelini eğitmek için kullanılan ana dosya.
    *   **`utils.py`**: LSTM eğitim süreci için yardımcı fonksiyonları içerir.
*   **`neural/`**: Genel sinir ağı tabanlı derin öğrenme modelleri için dosyaları içerir.
    *    **`EMGDataset.py`**: EMG veri kümesi yükleme ve ön işleme işlemleri için kullanılır.
    *   **`eval.py`**: Sinir ağı modelini değerlendirmek için kullanılan dosya.
     *   **`main.py`**: Sinir ağı eğitim sürecini başlatır.
    *   **`NeuralNetwork.py`**: Sinir ağı model mimarisini tanımlar.
    *   **`train.py`**: Sinir ağı modelini eğitmek için kullanılan ana dosya.
    *   **`utils.py`**: Sinir ağı eğitim süreci için yardımcı fonksiyonları içerir.
*   **`results/`**: Eğitilen modeller, performans metrikleri (accuracy, precision, recall, F1-score vb.), karışıklık matrisleri ve diğer görsel çıktılar bu klasörde saklanır.
*   **`requirements.txt`**: Proje için gerekli tüm Python paketlerinin listesini içerir.
*   **`README.md`**: Bu belge.

## Gereksinimler

Projenin çalışması için aşağıdaki yazılımların ve Python kütüphanelerinin kurulu olması gerekmektedir:

1.  **Python Sürümü**: 3.8 veya üstü
2.  **Python Kütüphaneleri**:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `imblearn`
    *   `seaborn`
    *   `matplotlib`
    *   `tqdm`
    *   `scipy`
    *   `torch`
    *   `tensorflow` (LSTM ve Neural klasörleri için)
    *   `xgboost` (boosting_code.py için)

### Bağımlılıkları Kurulumu

Gerekli tüm kütüphaneleri kurmak için aşağıdaki adımları izleyin:

1.  **Sanal Ortam Oluşturma**:
    ```bash
    python -m venv env
    ```

2.  **Sanal Ortamı Aktifleştirme**:
    *   **Linux/macOS**:
        ```bash
        source env/bin/activate
        ```
    *   **Windows**:
        ```bash
        env\Scripts\activate
        ```

3.  **Bağımlılıkları Yükleme**:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

Her bir dosya, belirli bir algoritma ile model eğitimi için tasarlanmıştır. Aşağıdaki adımlar takip edilerek çalıştırılabilir:

### Makine Öğrenimi Modellerini Çalıştırma

1.  **Karar Ağacı (Decision Tree)**:
    ```bash
    python decision_tree_trainer.py
    ```

2.  **Lojistik Regresyon (Logistic Regression)**:
    ```bash
    python log_reg_trainer.py
    ```

3.  **Random Forest**:
    ```bash
    python random_forest_trainer.py
    ```

4.  **SVM**:
    ```bash
    python svm_trainer.py
    ```

5.  **Boosting Algoritmaları ve Bagging**:
    ```bash
    python boosting_code.py
    ```

### Derin Öğrenme Modellerini Çalıştırma

1.  **LSTM**:
    *   **Eğitim**:
        ```bash
        python lstm/train.py
        ```
    *   **Değerlendirme**:
        ```bash
        python lstm/eval.py
        ```

2.  **Neural Network (Genel Sinir Ağları)**:
    *   **Eğitim**:
        ```bash
        python neural/train.py
        ```
    *   **Değerlendirme**:
        ```bash
        python neural/eval.py
        ```


## Sonuçlar

*   Eğitilen modeller, performans metrikleri ve görsel çıktılar `results` klasöründe saklanır.
*   Her modelin eğitim çıktısı, ilgili dosyaların çalıştırılması esnasında konsolda görüntülenecektir.
*   Model eğitim süreçleriyle ilgili ayrıntılı bilgiler, log dosyaları içerisinde yer almaktadır.
*   **results** klasörü içinde `model.pkl` veya `model.h5` gibi isimlerle model dosyaları kaydedilir.
*   Karışıklık matrisleri, performans grafikleri ve diğer görseller bu klasöre kaydedilecektir.
*   Eğitim sırasında elde edilen metrikler (accuracy, precision, recall, F1-score vb.) ve ilgili grafikler yine sonuçlar klasöründe tutulur.

## Ek Bilgiler

*   Her bir dosya (örneğin `decision_tree_trainer.py`) içindeki yorum satırları ve değişken adları, kodun anlaşılırlığını ve kullanım kolaylığını arttırmak için tasarlanmıştır.
*   Model eğitimleri sırasında, veri ön işleme adımları, model parametre ayarları, doğrulama (validation) teknikleri ve değerlendirme metrikleri gibi önemli detaylara yer verilmiştir.
*   Derin öğrenme dosyalarında, eğitim parametreleri (epoch sayısı, batch size, learning rate vb.) ve model mimarisi gibi detaylar ayarlanabilir.

Bu README dosyası, projenin anlaşılması ve etkili bir şekilde kullanılması için kapsamlı bir rehber sunmayı amaçlamaktadır. İyi çalışmalar!