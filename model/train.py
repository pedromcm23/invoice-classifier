import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from model.transformer import ColumnSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faturas_sinteticas.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.pkl")
def treinar_modelos():
    # Dataset Default
    df = pd.read_csv(DATA_PATH)
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
            #("num",  StandardScaler(), ["valor"]),
            ("num",  MinMaxScaler(), ["valor"]),
        ]
    )

    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    pipelines_treinados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Treinar e avaliar cada um
    for nome, clf in modelos.items():
        print(f"=== A avaliar: {nome} ===")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        print(f"Accuracy CV média: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        
        # Treinar no dataset completo
        pipe.fit(X, y)
        pipelines_treinados[nome] = pipe

    # Exportar o dicionário com todos os modelos estruturados
    joblib.dump(pipelines_treinados, MODEL_PATH)
    joblib.dump(categories, CATEGORIES_PATH)
    print(f"Modelo exportado para: {MODEL_PATH}")

if __name__ == "__main__":
    print("A iniciar o script de treino...")
    treinar_modelos()

'''
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
'''
