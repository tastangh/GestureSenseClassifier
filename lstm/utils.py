from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define plotting functions
def plot_loss(train_losses, val_losses, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(val_accuracies, val_f1_scores, num_epochs):
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Accuracy', color=color)
    ax1.plot(epochs, val_accuracies, 'b-', label='Val Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Validation F1 Score', color=color)
    ax2.plot(epochs, val_f1_scores, 'r-', label='Val F1 Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Validation Accuracy and F1 Score Over Epochs')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()