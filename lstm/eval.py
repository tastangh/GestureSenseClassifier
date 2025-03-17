import time, torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Define evaluation function
def evaluate_model(model, test_loader, device, testing_times):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    start_eval_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    end_eval_time = time.time()
    testing_times.append(end_eval_time - start_eval_time)

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')

    print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, '
          f'Test Recall: {test_recall:.4f}, Test Precision: {test_precision:.4f}')

    return all_labels, all_preds

