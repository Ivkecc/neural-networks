# Detekcija ljudskih aktivnosti primenom dubokih neuronskih mreža

Projekat iz oblasti dubokog učenja čiji je cilj klasifikacija ljudskih aktivnosti na osnovu podataka dobijenih sa akcelerometra i žiroskopa korišćenjem različitih arhitektura neuronskih mreža.

---

# Opis projekta

Cilj projekta je klasifikacija ljudskih aktivnosti korišćenjem senzorskih podataka prikupljenih sa pametnog telefona.

Implementirani modeli:

- Dense Neural Network (DenseNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- CNN + LSTM

Svi modeli koriste zajedničku infrastrukturu za treniranje, evaluaciju i vizualizaciju rezultata, što omogućava njihovo objektivno poređenje.

---

# Skup podataka

U projektu je korišćen **UCI Human Activity Recognition (HAR) Dataset**.

https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset/data?select=UCI-HAR+Dataset 

Skup podataka sadrži šest aktivnosti:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Korišćeni senzori:

- Akcelerometar (X, Y, Z)
- Žiroskop (X, Y, Z)

---

# Struktura projekta

```
src/
└── neural_networks/
    ├── config/
    ├── data/
    ├── datasets/
    ├── evaluation/
    ├── models/
    ├── training/
    └── visualization/

outputs/
plots/
checkpoints/
```

---

# Funkcionalnosti

Projekat obuhvata:

- implementaciju pet različitih arhitektura neuronskih mreža,
- zajednički trening pipeline za sve modele,
- Early Stopping,
- automatsko čuvanje najboljeg modela (Checkpointing),
- generisanje grafika Loss i Accuracy,
- generisanje Confusion Matrix,
- generisanje Classification Report-a.

---

# Tok izvršavanja

```
Skup podataka
        │
        ▼
DataLoader
        │
        ▼
Odabrani model
(DenseNN / CNN / LSTM / GRU / CNN+LSTM)
        │
        ▼
Treniranje
        │
        ▼
Evaluacija
        │
        ▼
Grafici i izveštaji
```

---

# Rezultati

| Model | Accuracy | Weighted F1-score |
|--------|----------:|------------------:|
| CNN | **89.11%** | **89.12%** |
| DenseNN | 86.87% | 86.94% |
| CNN + LSTM | 85.61% | 85.61% |
| GRU | 58.74% | 55.22% |
| LSTM | 38.92% | 29.04% |

Najbolje rezultate ostvario je **CNN model**, koji je pokazao najveću tačnost klasifikacije i najbolje ukupne performanse.

---

# Instalacija
```bash
git clone https://github.com/Ivkecc/neural-networks.git
cd neural-networks
```

Instalacija biblioteka

```bash
uv sync
```

ili

```bash
pip install -r requirements.txt
```


---

# Pokretanje projekta

Odabrati model u datoteci:

```
config/config.py
```

Primer:

```python
MODEL_TYPE = "cnn"
```

Pokretanje treninga:

```bash
python -m neural_networks.training.trainer
```

ili

```bash
uv run python -m neural_networks.training.trainer
```

---

# Generisani rezultati

Tokom treniranja automatski se generišu:

- grafici Accuracy i Loss,
- Confusion Matrix,
- Classification Report,
- najbolji model (checkpoint).

Rezultati se čuvaju u direktorijumima:

```
plots/
outputs/
checkpoints/
```

---

# Korišćene tehnologije

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

---

# Autori

- Luka Ivanković RA178/2023
- Marko Vasilić RA166/2023

Fakultet tehničkih nauka, Univerzitet u Novom Sadu
