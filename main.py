import pandas as pd
import torch
from src.dataset import EmotionDataset
from src.train import train_cross_validation
from src.evaluate import evaluate_model
import os
import random
import numpy as np

#reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#device
device = 'cpu'
print(f"Using device: {device}")

#load dataset
data_path = 'data/emotions_cleaned.csv'
df = pd.read_csv(data_path)

#shuffle dataset
df = df.sample(13000, random_state=SEED).reset_index(drop=True)

#hyperparameters
BATCH_SIZE = 128
MAX_LEN = 64
EPOCHS = 3
LR = 2e-5
N_SPLITS = 5

#train with cross-validation
print("Starting training with cross-validation...")
fold_results = train_cross_validation(
    df=df,
    n_splits=N_SPLITS,
    batch_size=BATCH_SIZE,
    max_len=MAX_LEN,
    epochs=EPOCHS,
    lr=LR,
    device=device
)

print("\n=== Cross-validation results ===")
for r in fold_results:
    print(f"Fold {r['fold']}: Val Accuracy={r['val_accuracy']:.4f}, Val F1={r['val_f1']:.4f}")

last_model_path = fold_results[-1]['model_path']

if os.path.exists(last_model_path):
    print("\nEvaluating last fold model...")
    acc, f1, cm = evaluate_model(
        last_model_path,
        df,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        device=device
    )
else:
    print(f"Model path does not exist: {last_model_path}")