# Domaći zadatak: Transformer - Emotion Detection in Text

Ovaj projekat implementira fine-tuning ***BERT*** modela (Bert Tiny, Bert Mini) za detekciju emocija u tekstu. Projekat koristi ***MLflow*** za praćenje eksperimenata, logovanje metrika (Accuracy, F1, itd.), čuvanje artefakata (Matrice konfuzije, ROC krive, Learning krive) i verzionisanje modela.

## Metodologija evaluacije
Kako bismo izbegli overfitting, kosristimo ***K-Fold Cross-Validation***

Proces funkcioniše na sledeći način:
1.  Ceo dataset se deli na **K** jednakih delova (fold-ova).
2.  Trening se pokreće **K** puta.
3.  U svakoj iteraciji, jedan fold se koristi za **validaciju**, a preostalih K-1 za **trening**.
4.  Konačni rezultati (Accuracy, F1, Loss) predstavljaju **prosek** svih K iteracija.

Random seed je hardcode-ovan na 69.
Verzija Python-a je 3.12.0

## Instalacija i podešavanje

Klonirati repozitorijum sa GitHub-a:
```
git clone https://github.com/uncle-draza/VIP_Domaci_Emotion_Detection.git
```

Instalirati dependency biblioteke:
```
pip install -r requirements.txt
```

## Struktura projekta
```
├── data                                # folder sa fajlovima
│    ├── processed                      # procesirani fajlovi
│    │   ├── emotions_cleaned.csv       # preprocesirani dataset
│    │   └── label_mapping.json         # izdvojene klase
│    └── raw                            # folder sa sirovim podacima
│       └── synthetic_emotions.csv      # sirovi dataset
├── mlruns                              # folder sa mlflow artifaktima i grafikonima
│   └── 1                               
│       └── ...
├── notebooks                           # folder sa notebookovima, vidi uputstvo
│    ├── 1_Analysis_and_Visualization.ipynb 
│    └── 2_Comparison_and_Report.ipynb                           
├── src                                 # folder sa glavnim kodom
│    ├── data_preprocessing.py          # modul za preprocesiranje
│    ├── dataset.py                     # modul za definisanje dataseta
│    ├── model.py                       # modul za definisanje arhitekture modela
│    └── train.py                       # glavni fajl za pokretanje treninga
├── .gitignore                          
├── mlflow.db                           # SQLite baza za cuvanje logova
└── README.md                           # ovo uputstvo
```

<pre>
NAPOMENA: Sadržaj data/processed foldera možete a i ne morate obrisati, svakako će se regenerisati
</pre>

## Uputstvo za korišćenje

### 1. Analiza i preprocesiranje
U folderu notebooks se nalazi notebook ***1_Analysis_and_Visualization.ipynb***. Njegovim pokretanjem se izvršava osnovna analiza podataka, kao i njihovo čišćenje tj. preprocesiranje i sama priprema za korišćenje.
Uz kod, nalaze se i komentari i zapažanja ovog procesa.

Pkretanjem ovog koda, primetićete da će se u folderu notebooks/processed pojaviti novi sadržan, očišćeni dataset, kao i izdvojene i enumerisane klase koje se u njemu pojavljuju.

Radnja preprocesiranja se može pokrenuti i nezavistno od notebooka prostim pokretanjem same python skripte.


### 2. Trening 
U folderu src, možete videti fajl train.py. Na samom kraju, možete videti ovaj kod:

<pre>
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
</pre>

U ovom delu koda možete podesiti same hiperparametre. O njima u daljem tekstu.
Pokretanjem ovog fajla komandom

```
python train.py
```
pokreće se sam proces treninga. U terminalu ispod se mogu videti informacije o toku treninga.

### Neke od konfiguracija parametara
Ove konfiguracije se mogu videti u MLFlow-u (o njemu u daljem tekstu), ali evo nekoliko konfiguracija koje možete probati za model bert-tiny:

***Najbolja preciznost:***
<pre>
    Learning rate (lr): 2e-05
    Batch size (batch_size): 32
    Max Length (max_len): 128
    Folds (n_splits): 4
    Epochs (epochs): 4
</pre>

***Brzo treniranje, manja tacnost:***
<pre>
    Learning rate (lr): 5e-05
    Batch size (batch_size): 64
    Max Length (max_len): 20
    Folds (n_splits): 2
    Epochs (epochs): 2 
</pre>

***Eksplozija, preveliki learning rate:***
<pre>
    Learning rate (lr): 0.01
    Batch size (batch_size): 16
    Max Length (max_len): 128
    Folds (n_splits): 3
    Epochs (epochs): 3
</pre>

### 3. Prikaz rezultata eksperimenta tj. run-a
Za praćenje rezultata eksperimenta, koristi se MLFlow koji se pokreće u terminalu komandom:
```
mlflow ui
```
Kao rezultat se dobija adresa: http://127.0.0.1:5000 na kojoj se može pristupiti korisničkom interfejsu.

U sekciji ***Experiments***, odabrati ***Emotion_Detection_Transformers*** i kliknuti na sekciju ***Runs***. U dobijenoj tabeli se vidi pregled svih eksperimenata sa određenim metrikama. Daljim klikom na odabrani run se mogu videti detaljnije metrike i grafikoni. Sačuvani grafikoni poput matrice konfuzije, ROC krive i Learning Rate krive se mogu videti u tabu ***Artifacts***. Ovi fajlovi se nalaze i u projekat folderu pod podfolderom ***mlruns/1***.


### 4. Prikaz izveštaja, poređenja i zapažanja
Unutar fodlera notebooks se nalazi notebook pod imenom ***2_Comparison_and_Report.ipynb***.
Pokretanjem njega se mogu dobiti poređenja odabranih eksperimenata sa opisom i zapažanjima.
Mogu se videti tabele sa sortiranim eksperimentima po accuracy i f1 score-u kao i poređenja grafikona matrice konfuzije, ROC krive i Learning Rate krive.