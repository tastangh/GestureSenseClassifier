from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, class_names=None):
        """
        Model değerlendirme sınıfını başlatır.
        
        :param class_names: Sınıf isimleri (opsiyonel, karmaşıklık matrisi için kullanılabilir).
        """
        self.class_names = class_names

    def evaluate(self, y_test, y_pred):
        """
        Modeli değerlendirir ve doğruluk skorunu yazdırır.
        
        :param y_test: Gerçek etiketler.
        :param y_pred: Tahmin edilen etiketler.
        """
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Doğruluk: {accuracy:.3f}\n")

        cm = metrics.confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, title="Karmaşıklık Matrisi")

    def plot_confusion_matrix(self, cm, title="Karmaşıklık Matrisi"):
        """
        Karmaşıklık matrisini çizer.
        
        :param cm: Karmaşıklık matrisi.
        :param title: Grafik başlığı.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names)) if self.class_names else np.arange(len(cm))
        plt.xticks(tick_marks, self.class_names, rotation=45) if self.class_names else plt.xticks(tick_marks)
        plt.yticks(tick_marks, self.class_names) if self.class_names else plt.yticks(tick_marks)

        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.tight_layout()
        plt.show()
