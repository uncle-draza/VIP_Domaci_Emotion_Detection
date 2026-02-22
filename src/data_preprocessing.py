import pandas as pd
import json
import os

class DatasetPreprocessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.label_mapping = {}

    def load_data(self):
        print(f"Ucitavanje podataka sa: {self.input_path}")
        return pd.read_csv(self.input_path)

    def clean_data(self, df):
        print("Ciscenje podataka...")
        initial_shape = df.shape
        
        # Uklanjanje duplikata iz teksta
        df = df.drop_duplicates(subset=['text'])
        print(f"Uklonjeno duplikata: {initial_shape[0] - df.shape[0]}")
        
        # Uklanjanje redova gde fali tekst ili emocija
        df = df.dropna(subset=['text', 'emotion'])

        return df

    def encode_labels(self, df):
        print("Enumeracija labela...")
        labels = df['emotion'].unique()
        self.label_mapping = {label: int(idx) for idx, label in enumerate(labels)}
        
        # Cuvanje mapiranje u JSON file
        mapping_path = os.path.join(self.output_dir, "label_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(self.label_mapping, f)
        print(f"Mapiranje labela sacuvano u: {mapping_path}")
        
        # Primena mapiranja
        df['label'] = df['emotion'].map(self.label_mapping)
        return df

    def run_pipeline(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        df = self.load_data()
        df = self.clean_data(df)
        df = self.encode_labels(df)
        
        # Cuvamo samo tekst i numericku labelu
        output_path = os.path.join(self.output_dir, "emotions_cleaned.csv")
        df[['text', 'label']].to_csv(output_path, index=False)
        print(f"Preprocesirani podaci sacuvani u: {output_path}")
        
        return df

# Preprocesiranje se moze pokrenuti i odavde (za potrebe testa itd.)
if __name__ == "__main__":
    preprocessor = DatasetPreprocessor(
        input_path="./data/raw/synthetic_emotions.csv",
        output_dir="./data/processed/"
    )
    preprocessor.run_pipeline()