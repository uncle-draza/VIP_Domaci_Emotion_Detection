import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import mlflow
import numpy as np
import random
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os
from datetime import datetime

from dataset import EmotionDataset
from model import EmotionClassifier

def set_seed(seed=69):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_epoch(model, dataloader, criterion, optimizer, device):
    #trenira model jednu epohu i vraca prosecan loss
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    #evaluacija modela, vraca accuracy i f1 score.
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), inference_time

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.ylabel('Stvarna klasa')
    plt.xlabel('Predvidjena klasa')
    plt.title('Confusion Matrix (Aggregated)')
    plt.close(fig)
    return fig

def plot_roc_curve(y_true, y_probs, n_classes, labels):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.close(fig)
    return fig

def plot_learning_curves(history, experiment_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for fold_label, losses in history.items():
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker='o', label=fold_label)
    
    ax.set_title(f'Learning Curve (Loss) - {experiment_name}')
    ax.set_xlabel('Epoha')
    ax.set_ylabel('Loss (Gubitak)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def run_experiment(df, experiment_base_name, class_names, n_splits=3, epochs=3, batch_size=16, lr=2e-5, max_len=128, description="N/A"):
    set_seed(69)
    device = torch.device('cpu') #ja koristim cpu, jer nemam dedicated gpu
    print(f"Device (cpu/gpu): {device}")

    n_classes = len(class_names)
    print(f"Prepoznate klase ({n_classes}): {class_names}")

    # dinamicko ime eksperimetna
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_base_name}_{timestamp}"
    print(f"Pokretanje eksperimenta: {experiment_name}")
    
    # priprema dataseta
    dataset = EmotionDataset(texts=df['text'].to_numpy(), labels=df['label'].to_numpy(), max_len=max_len)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=69)
    
    # podesavanje mlflow-a
    mlflow.set_experiment("Emotion_Detection_Transformers")
    
    with mlflow.start_run(run_name=experiment_name):
        # logovanje hiperparametara
        mlflow.log_params({
            "model": "bert-tiny",
            "n_splits": n_splits,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_len": max_len,
            "device": str(device),
            "description": description
        })
        
        overall_y_true = []
        overall_y_pred = []
        overall_y_probs = []
        training_times = []
        inference_times = []

        history = {}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
            
            fold_losses = []

            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
            
            model = EmotionClassifier(n_classes=n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            start_train = time.time()
            
            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                
                fold_losses.append(train_loss)

                # logovanje gubitka po epohi i foldu
                mlflow.log_metric(f"fold_{fold+1}_loss", train_loss, step=epoch)
                print(f"Epoha {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

            history[f"Fold {fold+1}"] = fold_losses

            train_time = time.time() - start_train
            training_times.append(train_time)
            
            y_true, y_pred, y_probs, inf_time = evaluate(model, val_loader, device)
            inference_times.append(inf_time)

            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)
            overall_y_probs.extend(y_probs)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)
            mlflow.log_metric(f"fold_{fold+1}_f1", f1)
            print(f"--> Accuracy: {acc:.4f}, Time per fold: {train_time:.1f}s")
            
        avg_acc = accuracy_score(overall_y_true, overall_y_pred)
        avg_f1 = f1_score(overall_y_true, overall_y_pred, average='macro')
        avg_train_time = np.mean(training_times)
        avg_inf_time = np.mean(inference_times)
        model_size_mb = get_model_size_mb(model)
        
        mlflow.log_metric("avg_accuracy", avg_acc)
        mlflow.log_metric("avg_f1", avg_f1)
        mlflow.log_metric("avg_train_time_sec", avg_train_time)
        mlflow.log_metric("avg_inference_time_sec", avg_inf_time)
        mlflow.log_metric("model_size_mb", model_size_mb)

        print(f"\n=== REZULTATI EKSPERIMENTA ===")
        print(f"Avg Accuracy: {avg_acc:.4f}")
        print(f"Avg Train Time Per Fold: {avg_train_time:.2f}s")
        print(f"Model Size: {model_size_mb:.2f} MB")

        fig_cm = plot_confusion_matrix(overall_y_true, overall_y_pred, class_names)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        
        fig_roc = plot_roc_curve(np.array(overall_y_true), np.array(overall_y_probs), n_classes=n_classes, labels=class_names)
        mlflow.log_figure(fig_roc, "roc_curve.png")

        fig_loss = plot_learning_curves(history, experiment_name)
        mlflow.log_figure(fig_loss, "learning_curve.png")
        
        print("Grafikoni sacuvani u MLflow!")

if __name__ == "__main__":
    df_cleaned = pd.read_csv("./data/processed/emotions_cleaned.csv")
    
    mapping_path = "./data/processed/label_mapping.json"

    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            label_mapping = json.load(f)

        sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1])
        class_names = [k for k, v in sorted_labels]
    else:
        print("label_mapping.json ne postoji, fallback na generic nazive klasa")
        class_names = ["Class_" + str(i) for i in range(7)]

    run_experiment(
        df=df_cleaned, 
        experiment_base_name="Bert_Tiny", 
        class_names=class_names,
        n_splits=4,
        epochs=4,
        batch_size=32,
        lr=2e-5,
        max_len=128,
        description="Optimalni hiperparametri za accuracy"
    )