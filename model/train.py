import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from model.transformer import ColumnSelector

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faturas_sinteticas.csv")
HISTORICO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "historico.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.pkl")

# Dataset Default
df = pd.read_csv(DATA_PATH)
# Correções do utilizador, caso existam
if os.path.exists(HISTORICO_PATH):
    with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
        historico_dados = json.load(f)
    
    if len(historico_dados) > 0:
        df_historico = pd.DataFrame(historico_dados)

        # Junta aos dados default
        df = pd.concat([df, df_historico[["fornecedor", "descricao", "valor", "categoria"]]], ignore_index=True)
        print(f"Adicionadas {len(df_historico)} faturas corrigidas pelo utilizador!")
print(f"Dataset carregado: {len(df)} registos, {df['categoria'].nunique()} categorias\n")

X = df[["fornecedor", "descricao", "valor"]]
y = df["categoria"]

categories = sorted(y.unique().tolist())
print(f"Categorias ({len(categories)}):")
for c in categories:
    print(f"  - {c}")
print()

# Pipeline separado para fornecedor — vocabulário pequeno e focado em nomes de empresa
forn_pipeline = Pipeline([
    ("sel",  ColumnSelector("fornecedor")),
    ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2),
                              min_df=1, sublinear_tf=True)),
])

# Pipeline separado para descrição — vocabulário maior para frases de fatura
desc_pipeline = Pipeline([
    ("sel",  ColumnSelector("descricao")),
    ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2),
                              sublinear_tf=True)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("forn", forn_pipeline, ["fornecedor", "descricao", "valor"]),
        ("desc", desc_pipeline, ["fornecedor", "descricao", "valor"]),
        ("num",  StandardScaler(), ["valor"]),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
])

print("A realizar cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"Accuracy CV média: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Scores por fold: {[f'{s:.4f}' for s in cv_scores]}\n")

print("A treinar modelo final em todos os dados...")
pipeline.fit(X, y)

y_pred = pipeline.predict(X)
print("Classification Report (treino completo):")
print(classification_report(y, y_pred, target_names=categories))

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(categories, CATEGORIES_PATH)
print(f"Modelo exportado para: {MODEL_PATH}")
print(f"Categorias exportadas para: {CATEGORIES_PATH}")
