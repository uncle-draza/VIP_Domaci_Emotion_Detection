import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, model_name="prajjwal1/bert-tiny", dropout_rate=0.3):

        super(EmotionClassifier, self).__init__()
        
        # ucitavnje modela
        self.bert = AutoModel.from_pretrained(model_name)
        
        # podesavanje dropout rate kako model ne bi ucio napamet
        self.drop = nn.Dropout(p=dropout_rate)
        
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        output = self.drop(pooled_output)
        
        return self.out(output)