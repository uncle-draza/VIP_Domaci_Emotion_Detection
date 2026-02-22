import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name="prajjwal1/bert-tiny", max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer_name="prajjwal1/bert-tiny", max_len=128, batch_size=16, shuffle=True):
    dataset = EmotionDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer_name=tokenizer_name,
        max_len=max_len
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)