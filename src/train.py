import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import numpy as np
import random
import pandas as pd
import os

from dataset import EmotionDataset
from model import EmotionClassifier

def set_seed(seed=69):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    return acc, f1

def run_experiment(df, experiment_name, n_splits=3, epochs=3, batch_size=16, lr=2e-5, max_len=128):
    set_seed(69)
    device = torch.device('cpu') #ja koristim cpu, jer nemam dedicated gpu
    print(f"Device (cpu/gpu): {device}")
    
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
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
            
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
            
            model = EmotionClassifier(n_classes=7).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # logovanje gubitka po epohi i foldu
                mlflow.log_metric(f"fold_{fold+1}_loss", train_loss, step=epoch)
                print(f"Epoha {epoch+1}/{epochs} | Loss: {train_loss:.4f}")
                
            # evaluacija na kraju folda
            val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"Fold {fold+1} Rezultati -> Accuracy: {val_acc:.4f} | F1-Macro: {val_f1:.4f}")
            
            # logovanje metrika po foldu
            mlflow.log_metric(f"fold_{fold+1}_accuracy", val_acc)
            mlflow.log_metric(f"fold_{fold+1}_f1", val_f1)
            
            fold_metrics.append((val_acc, val_f1))
            
        # racunanje proseka svih foldova
        avg_acc = np.mean([m[0] for m in fold_metrics])
        avg_f1 = np.mean([m[1] for m in fold_metrics])
        
        mlflow.log_metric("avg_cv_accuracy", avg_acc)
        mlflow.log_metric("avg_cv_f1", avg_f1)
        
        print(f"\n=== KRAJ EKSPERIMENTA ===")
        print(f"Average accuracy: {avg_acc:.4f} | Average F1: {avg_f1:.4f}")
        
        # cuvanje modela iz poslednjog folda u mlflow? 
        # mlflow.pytorch.log_model(model, "model")

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