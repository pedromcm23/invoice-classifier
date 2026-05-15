import os
import sys
import json
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.transformer import TextCombiner  # noqa: F401 — required for joblib unpickling

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "..", "model", "categories.pkl")
HISTORICO_PATH = os.path.join(BASE_DIR, "..", "data", "historico.json")

model = None
categories = None


def load_model():
    global model, categories
    if model is None:
        model = joblib.load(MODEL_PATH)
        categories = joblib.load(CATEGORIES_PATH)


def load_historico():
    if not os.path.exists(HISTORICO_PATH):
        return []
    with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_historico(entries):
    with open(HISTORICO_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    data = request.get_json() if request.is_json else None
    if data:
        fornecedor = data.get("fornecedor", "").strip()
        descricao = data.get("descricao", "").strip()
        valor_str = str(data.get("valor", "")).strip()
    else:
        fornecedor = request.form.get("fornecedor", "").strip()
        descricao = request.form.get("descricao", "").strip()
        valor_str = request.form.get("valor", "").strip()

    if not fornecedor or not descricao or not valor_str:
        return jsonify({"error": "Preenche todos os campos."}), 400

    try:
        valor = float(valor_str.replace(",", "."))
    except ValueError:
        return jsonify({"error": "Valor inválido."}), 400

    input_df = pd.DataFrame([{"fornecedor": fornecedor, "descricao": descricao, "valor": valor}])
    categoria = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    top3_idx = probas.argsort()[::-1][:3]
    top3 = [{"categoria": classes[i], "probabilidade": round(float(probas[i]) * 100, 1)} for i in top3_idx]

    main_idx = classes.index(categoria)
    confianca = round(float(probas[main_idx]) * 100, 1)

    return jsonify({"categoria": categoria, "confianca": confianca, "top3": top3})


@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum ficheiro enviado."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Apenas ficheiros PDF são suportados."}), 400

    try:
        import pdfplumber
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        texto = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                texto += (page.extract_text() or "") + "\n"
        os.unlink(tmp_path)

        linhas = [l.strip() for l in texto.splitlines() if l.strip()]

        # extrair valor — padrão: número com € ou EUR
        valor = ""
        for linha in linhas:
            m = re.search(r"(\d{1,6}[.,]\d{2})\s*€?", linha)
            if m:
                valor = m.group(1).replace(",", ".")
                break

        # extrair fornecedor — primeira linha com >= 3 palavras ou linha com "NIF"/"empresa"
        fornecedor = ""
        for linha in linhas[:10]:
            if len(linha.split()) >= 2 and not re.match(r"^\d", linha):
                fornecedor = linha
                break

        # extrair descrição — linha que mencione palavras chave de fatura
        descricao = ""
        keywords = ["fatura", "serviço", "serviços", "compra", "fornecimento",
                    "mensalidade", "referente", "pagamento", "descrição"]
        for linha in linhas:
            if any(k in linha.lower() for k in keywords):
                descricao = linha
                break
        if not descricao and len(linhas) > 2:
            descricao = linhas[2]

        return jsonify({"fornecedor": fornecedor, "valor": valor, "descricao": descricao, "texto": texto[:500]})

    except Exception as e:
        return jsonify({"error": f"Erro ao processar PDF: {str(e)}"}), 500


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Dados inválidos."}), 400
    entries = load_historico()
    entries.append(data)
    save_historico(entries)
    return jsonify({"ok": True, "total": len(entries)})


@app.route("/historico", methods=["GET"])
def historico():
    return jsonify(load_historico())


@app.route("/historico", methods=["DELETE"])
def limpar_historico():
    save_historico([])
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
