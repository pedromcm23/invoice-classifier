# Classificador de Faturas — Associação Desportiva / Escola de Andebol

Projeto de Machine Learning para classificação automática de faturas em categorias do plano de contas de uma associação desportiva.

## Categorias suportadas

| # | Categoria |
|---|-----------|
| 1 | Material Desportivo |
| 2 | Água Luz e Gás |
| 3 | Comunicações |
| 4 | Seguros |
| 5 | Alimentação e Bebidas |
| 6 | Transportes e Deslocações |
| 7 | Publicidade e Marketing |
| 8 | Serviços de Contabilidade |
| 9 | Equipamento Informático |
| 10 | Arbitragem e Competições |
| 11 | Manutenção e Reparações |

## Estrutura do projeto

```
Assign2/
├── data/
│   ├── generate_data.py       # Gerador de dados sintéticos
│   └── faturas_sinteticas.csv # Dataset gerado (600 registos)
├── model/
│   ├── train.py               # Treino do modelo
│   ├── model.pkl              # Modelo treinado (gerado)
│   └── categories.pkl         # Lista de categorias (gerado)
├── app/
│   ├── app.py                 # Aplicação Flask
│   └── templates/
│       └── index.html         # Interface web
├── requirements.txt
└── README.md
```

## Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Gerar dados sintéticos

```bash
python data/generate_data.py
```

Gera `data/faturas_sinteticas.csv` com 600 faturas sintéticas (~8% com ruído).

### 3. Treinar o modelo

```bash
python model/train.py
```

- Treina um `RandomForestClassifier` com pipeline TF-IDF + StandardScaler
- Imprime accuracy em cross-validation 5-fold e classification report
- Exporta `model/model.pkl` e `model/categories.pkl`

### 4. Correr a aplicação web

```bash
python app/app.py
```

Abre o browser em **[http://localhost:5001](http://localhost:5001)**.


Preenche o formulário com **fornecedor**, **valor** e **descrição** para obter a categoria prevista e a confiança do modelo.  
Em alternativa, faz **upload de uma fatura em PDF** — o sistema tenta extrair os campos automaticamente.

## Modelo & Pipeline

- **Features de texto**: `fornecedor + descricao` → `TfidfVectorizer(max_features=500, ngram_range=(1,2))`
- **Feature numérica**: `valor` → `StandardScaler`
- **Classificador**: `RandomForestClassifier(n_estimators=200, random_state=42)`
- **Avaliação**: Cross-validation estratificada 5-fold

## Tecnologias

- Python 3.x
- scikit-learn — pipeline ML
- Flask — aplicação web
- pandas / numpy — manipulação de dados
- joblib — serialização do modelo
- pdfplumber — extração de texto de faturas PDF
