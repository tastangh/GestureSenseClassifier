# save_results.py
import pandas as pd
import json
import os

def save_results(output_dir, model_name, model_params, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy, filter_params, feature_params, scaler_type, balancer_type, format='excel'):
    """
    Saves model results to a file with a specified format (Excel, CSV, or TXT).
    :param output_dir: Directory to save the results.
    :param model_name: Name of the model.
    :param model_params: Model parameters.
    :param train_loss: Training loss.
    :param train_accuracy: Training accuracy.
    :param val_loss: Validation loss.
    :param val_accuracy: Validation accuracy.
    :param test_accuracy: Test accuracy.
    :param filter_params: Filter parameters.
    :param feature_params: Feature parameters.
    :param scaler_type: Type of scaler used.
    :param balancer_type: Type of balancer used.
    :param format: Format of output file ('excel', 'csv', 'txt').
    """
    new_entry = {
        "Model Name": model_name,
        "Model Parameters": str(model_params),
        "Training Loss": train_loss,
        "Training Accuracy": train_accuracy,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "Filter Parameters": json.dumps(filter_params),
        "Feature Parameters": json.dumps(feature_params),
        "Scaler Type": scaler_type,
        "Balancer Type": balancer_type,
    }
    new_df = pd.DataFrame([new_entry])

    if format == 'excel':
       file_path = os.path.join(output_dir, f"model_results.xlsx")
    else :
       file_path = os.path.join(output_dir, f"model_results.{format}")

    try:
        if format == 'excel':
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_excel(file_path, index=False)
            else :
                new_df.to_excel(file_path, index=False)

        elif format == 'csv':
           if os.path.exists(file_path):
              existing_df = pd.read_csv(file_path)
              updated_df = pd.concat([existing_df, new_df], ignore_index=True)
              updated_df.to_csv(file_path, index=False)
           else :
              new_df.to_csv(file_path, index=False)
        elif format == 'txt':
           with open(file_path, "a") as f:
             f.write(f"Model Name: {model_name}\n")
             f.write(f"Model Parameters: {model_params}\n")
             f.write(f"Training Loss: {train_loss:.4f}\n")
             f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
             f.write(f"Validation Loss: {val_loss:.4f}\n")
             f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
             f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
             f.write(f"Filter Parameters: {filter_params}\n")
             f.write(f"Feature Parameters: {feature_params}\n")
             f.write(f"Scaler Type: {scaler_type}\n")
             f.write(f"Balancer Type: {balancer_type}\n")
             f.write("-" * 50 + "\n")
        else:
           raise ValueError("Invalid output file format selected.")
    except FileNotFoundError:
        if format == 'excel':
            new_df.to_excel(file_path, index=False)
        elif format == 'csv':
            new_df.to_csv(file_path, index=False)
        elif format == 'txt':
            with open(file_path, "a") as f:
             f.write(f"Model Name: {model_name}\n")
             f.write(f"Model Parameters: {model_params}\n")
             f.write(f"Training Loss: {train_loss:.4f}\n")
             f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
             f.write(f"Validation Loss: {val_loss:.4f}\n")
             f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
             f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
             f.write(f"Filter Parameters: {filter_params}\n")
             f.write(f"Feature Parameters: {feature_params}\n")
             f.write(f"Scaler Type: {scaler_type}\n")
             f.write(f"Balancer Type: {balancer_type}\n")
             f.write("-" * 50 + "\n")
        else:
             raise ValueError("Invalid output file format selected.")

    print(f"Sonuçlar '{file_path}' dosyasına kaydedildi.")