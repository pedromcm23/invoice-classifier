import os
import sys
import json
import re
import joblib
import pandas as pd
import subprocess
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.transformer import TextCombiner, ColumnSelector  # noqa: F401 — required for joblib unpickling

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
    
    previsoes_modelos = {}
    
    for nome_modelo, pipeline_modelo in model.items():
        categoria_prevista = pipeline_modelo.predict(input_df)[0]
        probas = pipeline_modelo.predict_proba(input_df)[0]
        classes = list(pipeline_modelo.classes_)
        
        todas_probas = {classes[i]: round(float(probas[i]) * 100, 1) for i in range(len(classes))}
        
        previsoes_modelos[nome_modelo] = {
            "categoria": categoria_prevista,
            "confianca": todas_probas.get(categoria_prevista, 0.0),
            "todas_probas": todas_probas  # Enviado para o frontend conseguir calcular o vencedor mais tarde
        }
    
    categoria_sugerida = previsoes_modelos["Random Forest"]["categoria"]

    return jsonify({
        "categoria": categoria_sugerida,
        "previsoes": previsoes_modelos
    })


FORNECEDORES_CONHECIDOS = [
    "EDP Comercial", "EDP", "Galp Energia", "Galp", "Endesa", "Iberdrola",
    "Águas de Portugal", "EPAL", "Naturgy",
    "NOS", "MEO", "Vodafone", "NOWO",
    "Fidelidade", "Allianz", "Zurich", "Generali", "Tranquilidade", "AXA", "Ageas",
    "Continente", "Pingo Doce", "Intermarché", "Lidl", "Mercadona", "Auchan",
    "BP", "Repsol", "CP Comboios", "Uber", "Hertz", "FlixBus",
    "Decathlon", "Sport Zone", "Nike", "Adidas", "Intersport", "Puma", "Asics",
    "PricewaterhouseCoopers", "PwC", "Deloitte", "BDO", "KPMG",
    "Worten", "Fnac", "Apple", "Dell", "HP", "Lenovo", "Samsung",
    "Federação Portuguesa de Andebol", "Fixando", "ManutençãoPro",
]

MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def extrair_fornecedor(texto):
    texto_lower = texto.lower()
    # fornecedores mais longos primeiro para evitar match parcial (ex: "EDP Comercial" antes de "EDP")
    for nome in sorted(FORNECEDORES_CONHECIDOS, key=len, reverse=True):
        if nome.lower() in texto_lower:
            return nome
    # fallback: primeira linha não-numérica com pelo menos 2 palavras
    for linha in [l.strip() for l in texto.splitlines() if l.strip()][:15]:
        if len(linha.split()) >= 2 and not re.match(r"^\d", linha):
            return linha
    return ""


def extrair_valor(texto):
    # Prioridade 1: "Total a pagar [texto opcional] €XX,XX" — NOS e outros
    m = re.search(r'Total a pagar[^\n€]*€\s*([\d]+[.,][\d]{2})', texto, re.IGNORECASE)
    if m:
        v = float(m.group(1).replace(',', '.'))
        if 5 <= v <= 5000:
            return str(round(v, 2))

    # Prioridade 2: "Total a pagar: XX,XX €" — variante com valor antes do €
    m = re.search(r'Total a pagar[^\n€]*?([\d]+[.,][\d]{2})\s*(?:€|EUR)', texto, re.IGNORECASE)
    if m:
        v = float(m.group(1).replace(',', '.'))
        if 5 <= v <= 5000:
            return str(round(v, 2))

    # Prioridade 3: "Quanto tenho a pagar? XX,XX €" — EDP
    m = re.search(r'Quanto tenho\s+a pagar\??\s*([\d]+[.,][\d]{2})\s*€', texto, re.IGNORECASE)
    if m:
        v = float(m.group(1).replace(',', '.'))
        if 5 <= v <= 5000:
            return str(round(v, 2))

    # Prioridade 4: outros padrões de total explícito
    outros = [
        r'Valor a pagar[^\n€]*€\s*([\d]+[.,][\d]{2})',
        r'Valor a pagar[^\n€]*?([\d]+[.,][\d]{2})\s*€',
        r'Montante total[^\n€]*€\s*([\d]+[.,][\d]{2})',
        r'Valor desta fatura com IVA\s*([\d]+[.,][\d]{2})',
        r'Total fatura[^\n€]*€\s*([\d]+[.,][\d]{2})',
        r'Valor total[^\n€]*€\s*([\d]+[.,][\d]{2})',
    ]
    for padrao in outros:
        m = re.search(padrao, texto, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(',', '.'))
            if 5 <= v <= 5000:
                return str(round(v, 2))

    # Fallback: primeiro valor €XX,XX ou XX,XX€ entre 5 e 5000
    for c in re.findall(r'(\d{1,6}[.,]\d{2})\s*€', texto):
        v = float(c.replace(',', '.'))
        if 5 <= v <= 5000:
            return str(round(v, 2))
    for c in re.findall(r'€\s*(\d{1,6}[.,]\d{2})', texto):
        v = float(c.replace(',', '.'))
        if 5 <= v <= 5000:
            return str(round(v, 2))

    return ""


def extrair_data(texto):
    # "15 de março de 2024" ou "15 de março 2024"
    m = re.search(
        r"(\d{1,2})\s+de\s+(" + "|".join(MESES_PT.keys()) + r")\s+(?:de\s+)?(\d{4})",
        texto, re.IGNORECASE
    )
    if m:
        dia, mes_str, ano = m.group(1), m.group(2).lower(), m.group(3)
        mes = MESES_PT.get(mes_str, 1)
        return f"{ano}-{mes:02d}-{int(dia):02d}"

    # DD/MM/YYYY ou DD-MM-YYYY
    m = re.search(r"(\d{2})[/\-](\d{2})[/\-](\d{4})", texto)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    return ""


MESES_NOMES = (
    "janeiro|fevereiro|março|abril|maio|junho|"
    "julho|agosto|setembro|outubro|novembro|dezembro"
)


def _limpar_periodo(conteudo):
    """Remove números de fatura e datas DD-MM-YYYY que pdfplumber concatena na linha."""
    conteudo = re.sub(r'\bFT\s+\d+/\d+\b', '', conteudo, flags=re.IGNORECASE)
    conteudo = re.sub(r'\d{2}[-/]\d{2}[-/]\d{4}', '', conteudo)
    return re.sub(r'\s+', ' ', conteudo).strip()


def extrair_descricao(texto):
    # 1. "Período de faturação …" — EDP (intervalo) vs NOS (nome do mês)
    m = re.search(r"Per[ií]odo de fatura[çc][aã]o[:\s]+(.+)", texto, re.IGNORECASE)
    if m:
        conteudo = _limpar_periodo(m.group(1))
        # EDP: intervalo de datas "DD de mês a DD de mês"
        if re.search(r'\d{1,2}\s+de\s+\w+\s+a\s+\d{1,2}\s+de\s+\w+', conteudo, re.IGNORECASE):
            return "Período de faturação: " + conteudo[:80]
        # NOS/outros: sobra apenas o nome do mês — normaliza para "Mês de X"
        m_mes = re.search(r'(' + MESES_NOMES + r')(?:\s+de\s+\d{4}|\s+\d{4})?',
                          conteudo, re.IGNORECASE)
        if m_mes:
            return "Mês de " + m_mes.group(0).strip()
        if conteudo:
            return "Período de faturação: " + conteudo[:80]

    # 2. "Mês de Janeiro 2024" explícito no texto
    m = re.search(r"(M[eê]s de \w+(?:\s+\d{4})?)", texto, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 3. keywords úteis, excluindo linhas com número de fatura
    keywords = ["serviço", "serviços", "compra", "fornecimento", "mensalidade",
                "referente", "pagamento", "descrição", "energia", "eletricidade",
                "electricidade", "gás", "internet", "seguro", "telecomunicações"]
    for linha in [l.strip() for l in texto.splitlines() if l.strip()]:
        if re.search(r"\bFT\s+\d+/\d+\b", linha, re.IGNORECASE):
            continue
        if any(k in linha.lower() for k in keywords) and len(linha) > 8:
            return linha[:120]

    # 4. fallback: terceira linha não-numérica sem número de fatura
    linhas = [l.strip() for l in texto.splitlines()
              if l.strip() and not re.search(r"\bFT\s+\d+/\d+\b", l)]
    return linhas[2] if len(linhas) > 2 else ""


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

        fornecedor = extrair_fornecedor(texto)
        valor      = extrair_valor(texto)
        data       = extrair_data(texto)
        descricao  = extrair_descricao(texto)

        return jsonify({
            "fornecedor": fornecedor,
            "valor": valor,
            "data": data,
            "descricao": descricao,
            "texto": texto[:600],
        })

    except Exception as e:
        return jsonify({"error": f"Erro ao processar PDF: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
