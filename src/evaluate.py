import mlflow
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .dataset import EmotionDataset
from .model import EmotionClassifier

mlflow.set_tracking_uri("file:./mlruns")

def evaluate_model(model_path, df, max_len=128, batch_size=64, device='cpu'):
    dataset = EmotionDataset(
        texts=df['text'].tolist(),
        labels=df['labels'].tolist(),
        max_len=max_len
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #ucitavanje modela
    model = EmotionClassifier(n_classes=df['labels'].nunique()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #metrike
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # log u mlflow
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_artifact(model_path, artifact_path="model")

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

    return acc, f1, cm