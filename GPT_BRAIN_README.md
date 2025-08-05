# HiveGPT Brain

Un agente AI basato su GPT per il gioco Hive, ispirato all'approccio ALLIE per gli scacchi ma adattato per Hive.

## Panoramica

Questo modulo implementa un "brain" AI che utilizza un modello Transformer decoder-only (basato su GPT-2) per apprendere strategie di gioco Hive attraverso il fine-tuning su dati di partite. L'architettura è ispirata al paper ALLIE ma semplificata rimuovendo la componente temporale (pondering time).

### Architettura del Modello

- **Backbone**: GPT-2 medium (355M parametri) con embedding personalizzati per Hive
- **Policy Head**: Predice la distribuzione di probabilità sulle mosse possibili
- **Value Head**: Valuta la posizione corrente nel range [-1, 1]
- **Training**: Minimizza la log-likelihood delle mosse e l'errore quadratico medio della valutazione

### Funzionalità Principali

1. **Encoder di Gioco**: Converte sequenze di mosse Hive in token per l'addestramento
2. **Modello GPT**: Architettura Transformer con heads per policy e value
3. **Brain AI**: Integrazione nell'engine Hive esistente
4. **Training System**: Sistema completo per il fine-tuning su dati di partite

## Installazione

### Dipendenze

```bash
pip install torch>=2.0.0 transformers>=4.30.0 numpy>=1.21.0 datasets>=2.12.0 accelerate>=0.20.0
```

### Setup Completo

```bash
# Clona il repository
git clone https://github.com/Crystal-Spider/hivemind.git
cd hivemind

# Installa le dipendenze
pip install -r requirements.txt
```

## Utilizzo

### 1. Preparazione dei Dati

#### Formato Dati di Training

I dati devono essere in formato JSON con la seguente struttura:

```json
[
  {
    "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/"],
    "result": "white"
  },
  {
    "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\"],
    "result": "draw"
  }
]
```

#### Generazione di Dati di Esempio

```python
# Crea dati di esempio per test
python data_utils.py sample --output sample_games.json

# Analizza un dataset esistente
python data_utils.py analyze sample_games.json

# Converte gamestring in formato training
python data_utils.py process gamestrings.txt training_data.json
```

### 2. Addestramento del Modello

#### Addestramento Base

```python
from ai.gpt_brain import HiveGPTTrainer, load_game_data

# Carica i dati
games_data = load_game_data("training_data.json")

# Inizializza il trainer
trainer = HiveGPTTrainer(model_save_path="hive_gpt_model.pt")

# Addestra il modello
trainer.train(
    games_data=games_data,
    num_epochs=5,
    batch_size=8,
    learning_rate=5e-5
)
```

#### Script di Training

```bash
# Training con dati di esempio
python train_gpt.py --use-sample-data --epochs 3 --batch-size 4

# Training con dati personalizzati
python train_gpt.py --data-file my_games.json --epochs 5 --batch-size 8
```

### 3. Utilizzo nell'Engine

#### Integrazione Diretta

```python
from ai.gpt_brain import GPTBrain
from core.board import Board

# Inizializza il brain (carica automaticamente il modello se disponibile)
gpt_brain = GPTBrain(model_path="hive_gpt_model.pt")

# Crea una posizione di gioco
board = Board()
board.play("wS1")
board.play("bS1 wS1/")

# Trova la mossa migliore
best_move = gpt_brain.find_best_move(board, max_branching_factor=10)
print(f"Mossa suggerita: {best_move}")

# Valuta la posizione
evaluation = gpt_brain.evaluate_position(board)
print(f"Valutazione: {evaluation:.3f}")
```

#### Esempio Completo

```bash
python example_gpt_usage.py
```

## Architettura Tecnica

### Encoder di Gioco

Il `HiveGameEncoder` converte le sequenze di mosse Hive in token:

- **Token Speciali**: `<GAME_START>`, `<MOVE>`, `<WHITE>`, `<BLACK>`, etc.
- **Vocabolario Mosse**: Tutti i possibili pezzi e posizioni Hive
- **Sequenza**: `<GAME_START> <WHITE> <MOVE> wS1 <BLACK> <MOVE> bS1 wS1/ ...`

### Modello GPT

```python
class HiveGPTModel(nn.Module):
    def __init__(self, config, vocab_size):
        self.transformer = GPT2Model(config)  # Backbone
        self.policy_head = nn.Linear(config.hidden_size, vocab_size)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()
        )
```

### Loss Function

```
L(θ) = Σ[-log p_θ(m_i | m_{<i}) + (v_θ(m_{<i}) - v)²]
```

Dove:
- `p_θ(m_i | m_{<i})`: Probabilità della mossa i data la storia
- `v_θ(m_{<i})`: Valutazione della posizione
- `v`: Risultato effettivo della partita

## Personalizzazione

### Configurazione del Modello

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=encoder.vocab_size,
    n_positions=1024,      # Lunghezza massima sequenza
    n_embd=768,            # Dimensione hidden
    n_layer=12,            # Numero di layer
    n_head=12,             # Numero attention head
    n_inner=3072,          # Dimensione FFN
)
```

### Parametri di Training

```python
trainer.train(
    games_data=games_data,
    output_dir="./output",
    num_epochs=5,           # Epoche di training
    batch_size=8,           # Dimensione batch
    learning_rate=5e-5,     # Learning rate
    warmup_steps=500        # Step di warmup
)
```

## Prestazioni e Ottimizzazioni

### Requisiti Hardware

- **Minimo**: CPU con 8GB RAM per inferenza
- **Consigliato**: GPU con 8GB VRAM per training efficiente
- **Training Completo**: GPU con 16GB+ VRAM per batch size grandi

### Ottimizzazioni

1. **Gradient Checkpointing**: Per ridurre l'uso di memoria
2. **Mixed Precision**: Training in FP16 per velocità
3. **Model Parallelism**: Per modelli molto grandi

## Confronto con Altri Agenti

| Agente | Tipo | Punti di Forza | Punti di Debolezza |
|--------|------|---------------|-------------------|
| Random | Baseline | Veloce, semplice | Nessuna strategia |
| AlphaBeta | Tree Search | Logica forte, deterministico | Limitato da funzione di valutazione |
| GPT | Neural Network | Apprende pattern complessi | Richiede dati di training |

## Troubleshooting

### Errori Comuni

1. **Import Error**: Installare le dipendenze PyTorch e Transformers
2. **CUDA Out of Memory**: Ridurre batch_size o usare gradient_checkpointing
3. **Convergenza Lenta**: Aumentare learning_rate o aggiungere più dati

### Debug

```python
# Verifica il vocabolario
encoder = HiveGameEncoder()
print(f"Vocabolario: {encoder.vocab_size} token")

# Test encoding/decoding
sequence = encoder.encode_game_sequence(["wS1", "bS1 wS1/"], "draw")
print(f"Sequenza codificata: {sequence}")
```

## Contributi

Per contribuire al progetto:

1. Fork del repository
2. Crea un branch per la feature
3. Implementa modifiche con test
4. Submetti una pull request

## Licenza

Vedere LICENSE nel repository principale.

## Riconoscimenti

- Ispirato al paper ALLIE per l'architettura generale
- Basato su GPT-2 di OpenAI per il backbone Transformer
- Integrato nell'engine Hive esistente di Crystal-Spider
