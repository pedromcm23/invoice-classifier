import os
import sys
import json
import joblib
import pandas as pd
import subprocess
from google import genai
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.transformer import TextCombiner, ColumnSelector  # noqa: F401

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "..", "model", "categories.pkl")
HISTORICO_PATH = os.path.join(BASE_DIR, "..", "data", "historico.json")

model = None
categories = None

client = genai.Client(api_key="AIzaSyAY6gCxQrOVq3xBZBzqEoLpRRXP0_nBb0g")

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


def extrair_dados_com_ia(texto_sujo):
    prompt = f"""
    És um assistente financeiro especialista em ler faturas portuguesas.
    Analisa o seguinte texto extraído de um PDF (que pode estar baralhado ou ter caracteres estranhos) 
    e devolve APENAS um objeto JSON válido, sem mais nenhum texto.

    O JSON deve ter exatamente estas chaves:
    "fornecedor": O nome da empresa que emitiu a fatura (ex: "EDP Comercial", "MEO", "BOOMFIT").
    "valor": O valor total final a pagar, formatado apenas com números e ponto (ex: "136.17").
    "data": A data de emissão ou da fatura no formato YYYY-MM-DD.
    "descricao": Um resumo muito curto do que foi cobrado (ex: "Mensalidade TV e Internet", "Eletricidade e Gás", "Equipamento Desportivo").

    Texto da fatura:
    {texto_sujo[:3000]} 
    """
    
    try:
        resposta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        # Limpar os marcadores markdown que o LLM às vezes devolve
        texto_json = resposta.text.replace("```json", "").replace("```", "").strip()
        dados = json.loads(texto_json)
        return dados
    except Exception as e:
        print(f"[ERRO LLM]: {e}")
        return {"fornecedor": "", "valor": "", "data": "", "descricao": ""}

# ROTAS 

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
    
    print("\n=== ANÁLISE DE CONFIANÇA DOS ALGORITMOS ===")
    for nome_modelo, pipeline_modelo in model.items():
        categoria_prevista = pipeline_modelo.predict(input_df)[0]
        probas = pipeline_modelo.predict_proba(input_df)[0]
        classes = list(pipeline_modelo.classes_)
        todas_probas = {classes[i]: round(float(probas[i]) * 100, 1) for i in range(len(classes))}
        
        confianca = todas_probas.get(categoria_prevista, 0.0)
        print(f" -> [{nome_modelo}]: Sugere '{categoria_prevista}' com {confianca}% de certeza.")
        
        previsoes_modelos[nome_modelo] = {
            "categoria": categoria_prevista,
            "confianca": confianca,
            "todas_probas": todas_probas
        }
    print("===========================================\n")
    
    return jsonify({
        "categoria": previsoes_modelos["Random Forest"]["categoria"],
        "previsoes": previsoes_modelos
    })

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
                extraido = page.extract_text()
                if extraido:
                    texto += extraido + "\n"
        os.unlink(tmp_path)
        
        print("\n=== INÍCIO DA EXTRAÇÃO PDF ===")
        print(f"A analisar o ficheiro: {file.filename}...")
        
        # Chama a inteligência artificial para extrair os dados
        dados_extraidos = extrair_dados_com_ia(texto)
        
        print(f"Fornecedor extraído: -> {dados_extraidos.get('fornecedor')} <-")
        print(f"Valor extraído: -> {dados_extraidos.get('valor')} <-")
        print(f"Data extraída: -> {dados_extraidos.get('data')} <-")
        print(f"Descrição extraída: -> {dados_extraidos.get('descricao')} <-")
        print("========================================\n")

        return jsonify({
            "fornecedor": dados_extraidos.get("fornecedor", ""),
            "valor": dados_extraidos.get("valor", ""),
            "data": dados_extraidos.get("data", ""),
            "descricao": dados_extraidos.get("descricao", ""),
            "texto": texto[:600],
        })

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
    
    total_faturas = len(entries)
    if total_faturas > 0 and total_faturas % 10 == 0:
        try:
            print(f"\n[SISTEMA] {total_faturas} faturas atingidas! A iniciar re-treino em lote automático...")
            caminho_raiz = os.path.abspath(os.path.join(BASE_DIR, ".."))
            
            subprocess.run([sys.executable, "-m", "model.train"], cwd=caminho_raiz, check=True)
            
            global model, categories
            model = None
            categories = None
            print("[SISTEMA] Modelos atualizados com sucesso após lote de 10 faturas!\n")
        except Exception as e:
            print(f"[ERRO NO RE-TREINO EM LOTE]: {str(e)}")
            
    return jsonify({"ok": True, "total": total_faturas})

@app.route("/historico", methods=["GET"])
def historico():
    return jsonify(load_historico())

@app.route("/historico", methods=["DELETE"])
def limpar_historico():
    save_historico([])
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, port=5001)