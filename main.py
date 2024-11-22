# import the necessary packages
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from dataset import DataProcessor

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Spectral):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def random_forest_classification(X_train, X_test, y_train, y_test, classes):
    """Random Forest Sınıflandırma"""

    # Train the model using the training sets
    print("Randomforest Eğitim Oturumları Başladı..\n")

    RFclf = RandomForestClassifier(n_estimators=100)
    RFclf.fit(X_train, y_train)

    # Predict from Test set
    y_pred = RFclf.predict(X_test)

    # Compare prediction and actual class with accuracy score
    print("Accuracy : {accuracy:.3f}\n".format(accuracy=metrics.accuracy_score(y_pred, y_test)))
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plot_confusion_matrix(cm, classes=classes, title="Confusion Matrix: RandomForest Model")
    # fig.savefig("Resources/cmRF.png", dpi=100)
    plt.show()


def decoder(y_list):
    """One-hot Decoder Specified for LSTM Classification"""

    y_classes = []
    for el in y_list:
        y_classes.append(np.argmax(el))
    return np.array(y_classes)


def lstm_model(n_steps, n_features):
    """LSTM Model Definition"""
    model = Sequential()
    model.add(
        LSTM(50, return_sequences=True, input_shape=(n_steps, n_features)))

    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=64))
    model.add(Dense(units=128))

    model.add(Dense(units=4, activation="softmax"))
    model.compile(optimizer='adam', loss='mse')

    return model
def lstm_classification(X_train, X_test, y_train, y_test, saved_model=False):
    """LSTM Classification"""
    # Özellikleri yeniden şekillendir (8x8 matrise)
    X_train = X_train.values.reshape(-1, 8, 8)
    X_test = X_test.values.reshape(-1, 8, 8)

    # Etiketleri One-Hot Encoding'e çevir
    y_train = np.eye(np.max(y_train) + 1)[y_train]
    y_test = np.eye(np.max(y_test) + 1)[y_test]

    # Model yükle veya eğit
    if saved_model:
        print("Trained LSTM Model is loading...\n")
        model = load_model("results/lstm_model.h5")
    else:
        print("LSTM Training Session has begun...\n")
        model = lstm_model(8, 8)  # 8 sensör ve 8 zaman adımı
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

        # Kayıp grafiğini görselleştir
        plt.plot(history.history['loss'])
        plt.title('Model Kaybı')
        plt.ylabel('Kayıp')
        plt.xlabel('Epok')
        plt.legend(['Eğitim'], loc='upper left')
        plt.show()

        # Eğitilmiş modeli kaydet
        model.save("results/lstm_model.h5")
        print("Model saved to disk.\n")

    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Modelin doğruluğunu hesapla
    print("LSTM Accuracy: {:.3f}\n".format(metrics.accuracy_score(y_test_classes, y_pred_classes)))

    # Confusion Matrix'i çiz
    cm = metrics.confusion_matrix(y_test_classes, y_pred_classes)
    plot_confusion_matrix(cm, classes=["Taş", "Kağıt", "Makas", "OK"], title="Confusion Matrix: LSTM")


if __name__ == '__main__':

    data_path="dataset/"
    classes = ["Taş(0)", "Kağıt(1)", "Makas(2)", "OK(3)"]
    data_processor = DataProcessor(data_path, classes)
    dataset= data_processor.load_data()
    X = dataset.iloc[:, :-1]  # Özellik sütunları
    y = dataset.iloc[:, -1]   # Etiket sütunu

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = data_processor.train_test_split(X, y)
    data_processor.check_train_test_distribution(y_train, y_test)

    print("Main Function is running..\n")

    random_forest_classification(X_train, X_test, y_train, y_test,classes)

    lstm_classification(X_train, X_test, y_train, y_test, saved_model=False)




