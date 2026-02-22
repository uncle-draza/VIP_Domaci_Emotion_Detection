import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, max_len=128, model_name='distilbert-base-uncased'):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_dataloaders(df, batch_size=64, max_len=128, test_size=0.2, val_size=0.1, random_state=42):

    #izmesaj dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df['label']
    )

    train_loader = DataLoader(
        EmotionDataset(train_df['text'].tolist(), train_df['label'].tolist(), max_len=max_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        EmotionDataset(val_df['text'].tolist(), val_df['label'].tolist(), max_len=max_len),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        EmotionDataset(test_df['text'].tolist(), test_df['label'].tolist(), max_len=max_len),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader