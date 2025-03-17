
from train import clean_data, read_data, normalize_data, split_data, create_datasets, create_data_loaders, train_model
from utils import plot_loss, plot_metrics, plot_confusion_matrix
import torch
from LSTM_Model import LSTM_Model
from eval import evaluate_model
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    
    # Data operations
    
    file_path = "data.csv"
    data = read_data(file_path=file_path)
    X, y = clean_data(data)
    X = normalize_data(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    train_dataset, val_dataset, test_dataset = create_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Hyperparameters
    hidden_size = 128
    num_layers = 3
    dropout_rate = 0.4
    learning_rate = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device - ", device)

    # Create the model, define the loss function, and optimizer
    model = LSTM_Model(input_size=1, hidden_size=hidden_size, output_size=len(y.unique()), num_layers=num_layers,dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    training_times = []
    testing_times = []
    num_epochs = 5

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
    plot_loss(train_losses, val_losses, num_epochs)
    plot_metrics(val_accuracies, val_f1_scores, num_epochs)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate the model on the test set and get predictions
    y_true, y_pred = evaluate_model(model, test_loader, device = device, testing_times = testing_times)

    # Get class names (assuming y has categorical labels)
    class_names = sorted(y.unique())  # Ensure consistent ordering

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=False,
                        title='Confusion Matrix for Test Set')