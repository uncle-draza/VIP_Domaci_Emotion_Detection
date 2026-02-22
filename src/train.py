import os
import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from .dataset import EmotionDataset
from .model import EmotionClassifier
from torch.utils.data import DataLoader, Subset

mlflow.set_tracking_uri("file:./mlruns")


def train_fold(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        #validacija
        model.eval()
        val_loss = 0
        preds = []
        trues = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average='macro')

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_model_state


def train_cross_validation(df, n_splits=5, batch_size=64, max_len=128, epochs=3, lr=2e-5, device='cpu'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X = df['text'].tolist()
    y = df['label'].tolist()

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")

        dataset = EmotionDataset(X, y, max_len=max_len)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = EmotionClassifier(n_classes=len(set(y))).to(device)

        with mlflow.start_run(run_name=f"fold_{fold}"):
            mlflow.log_param("fold", fold)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lr", lr)
            mlflow.log_param("max_len", max_len)

            best_state = train_fold(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
            model.load_state_dict(best_state)

            mlflow.pytorch.log_model(model, f"model_fold_{fold}")

            os.makedirs("models", exist_ok=True)
            local_model_path = f"models/model_fold_{fold}.pth"
            torch.save(model.state_dict(), local_model_path)
            print(f"Saved local model to: {local_model_path}")

            model.eval()
            preds = []
            trues = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask)
                    preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    trues.extend(labels.cpu().numpy())

            val_acc = accuracy_score(trues, preds)
            val_f1 = f1_score(trues, preds, average='macro')

            fold_results.append({
                'fold': fold,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'model_path': local_model_path
            })

            mlflow.log_metric("final_val_accuracy", val_acc)
            mlflow.log_metric("final_val_f1", val_f1)

    return fold_results