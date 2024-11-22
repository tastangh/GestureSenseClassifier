from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

class ModelEvaluator:
    def __init__(self, class_names=None, save_path="results"):
        """
        Model değerlendirme sınıfını başlatır.
        
        :param class_names: Sınıf isimleri (opsiyonel, karmaşıklık matrisi için kullanılabilir).
        :param save_path: Grafiklerin kaydedileceği dizin.
        """
        self.class_names = class_names
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def evaluate(self, y_test, y_pred, model_name="Model"):
        """
        Modeli değerlendirir ve doğruluk skorunu yazdırır.
        
        :param y_test: Gerçek etiketler.
        :param y_pred: Tahmin edilen etiketler.
        :param model_name: Modelin adı (grafik kaydı için).
        """
        # Doğruluk
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"{model_name} Doğruluk: {accuracy:.3f}\n")

        # Confusion Matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name=model_name)

        # Grafik kaydetme
        save_file = os.path.join(self.save_path, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.savefig(save_file)
        print(f"{model_name} Confusion Matrix grafiği kaydedildi: {save_file}")

    def plot_confusion_matrix(self, cm, model_name="Model"):
        """
        Karmaşıklık matrisini çizer.
        
        :param cm: Karmaşıklık matrisi.
        :param model_name: Model adı (grafik başlığı).
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{model_name} Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(self.class_names)) if self.class_names else np.arange(len(cm))
        plt.xticks(tick_marks, self.class_names, rotation=45) if self.class_names else plt.xticks(tick_marks)
        plt.yticks(tick_marks, self.class_names) if self.class_names else plt.yticks(tick_marks)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]:d}", 
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen Etiket')
        plt.tight_layout()
        plt.show()
