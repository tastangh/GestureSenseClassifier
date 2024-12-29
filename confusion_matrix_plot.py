import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    @staticmethod
    def plot(y_true, y_pred, labels, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
