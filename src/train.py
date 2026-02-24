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
            _, preds = torch.max(outputs, dim=1)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), inference_time

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
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

def run_experiment(df, experiment_name, n_splits=3, epochs=3, batch_size=16, lr=2e-5, max_len=128):
    set_seed(69)
    device = torch.device('cpu') #ja koristim cpu, jer nemam dedicated gpu
    print(f"Device (cpu/gpu): {device}")

    class_names = ["Love", "Sad", "Anger", "Fun", "Hate", "Surprise", "Happiness"]
    
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
            "device": str(device)
        })
        
        overall_y_true = []
        overall_y_pred = []
        overall_y_probs = []
        training_times = []
        inference_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
            
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
            
            model = EmotionClassifier(n_classes=7).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            start_train = time.time()
            
            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # logovanje gubitka po epohi i foldu
                mlflow.log_metric(f"fold_{fold+1}_loss", train_loss, step=epoch)
                print(f"Epoha {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

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
            print(f"  --> Acc: {acc:.4f}, Time: {train_time:.1f}s")
            
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
        print(f"Avg Train Time: {avg_train_time:.2f}s")
        print(f"Model Size: {model_size_mb:.2f} MB")

        fig_cm = plot_confusion_matrix(overall_y_true, overall_y_pred, class_names)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        
        fig_roc = plot_roc_curve(np.array(overall_y_true), np.array(overall_y_probs), n_classes=7, labels=class_names)
        mlflow.log_figure(fig_roc, "roc_curve.png")
        
        print("Grafikoni sacuvani u MLflow!")

if __name__ == "__main__":
    # ucitavanje ociscenog dataseta
    df_cleaned = pd.read_csv("./data/processed/emotions_cleaned.csv")
    
    run_experiment(
        df=df_cleaned, 
        experiment_name="Baseline_Bert_Tiny_LR_2e5",
        n_splits=3,
        epochs=3,
        batch_size=16,
        lr=2e-5
    )