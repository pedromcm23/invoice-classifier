import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "..", "model", "categories.pkl")

model = None
categories = None


def load_model():
    global model, categories
    if model is None:
        model = joblib.load(MODEL_PATH)
        categories = joblib.load(CATEGORIES_PATH)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    fornecedor = request.form.get("fornecedor", "").strip()
    descricao = request.form.get("descricao", "").strip()
    valor_str = request.form.get("valor", "").strip()

    if not fornecedor or not descricao or not valor_str:
        return render_template("index.html", result=None, error="Preenche todos os campos.")

    try:
        valor = float(valor_str.replace(",", "."))
    except ValueError:
        return render_template("index.html", result=None, error="Valor inválido. Usa formato numérico (ex: 125.50).")

    input_df = pd.DataFrame([{"fornecedor": fornecedor, "descricao": descricao, "valor": valor}])
    categoria = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    idx = list(model.classes_).index(categoria)
    confianca = round(proba[idx] * 100, 1)

    result = {"categoria": categoria, "confianca": confianca}
    return render_template("index.html", result=result, error=None,
                           fornecedor=fornecedor, valor=valor_str, descricao=descricao)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
