import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Material Desportivo": {
        "fornecedores": ["Decathlon", "Sport Zone", "Nike", "Adidas", "Intersport", "Puma", "Asics"],
        "valor_min": 50, "valor_max": 800,
        "descricoes": [
            "Compra de bolas de andebol",
            "Equipamento de treino para atletas",
            "Coletes e fatos de treino",
            "Aquisição de redes para baliza",
            "Compra de sacos de desporto e mochilas",
            "Material de proteção e luvas de guarda-redes",
            "Sapatos e sapatilhas de andebol",
        ],
    },
    "Água Luz e Gás": {
        "fornecedores": ["EDP Comercial", "EDP", "Galp Energia", "Endesa Energia", "Endesa", "Iberdrola Portugal", "Iberdrola", "Águas de Portugal", "EPAL", "Naturgy"],
        "valor_min": 30, "valor_max": 300,
        "descricoes": [
            "Fatura de eletricidade mês de março",
            "Consumo de água instalações desportivas",
            "Fatura de gás aquecimento balneários",
            "Fornecimento de energia elétrica pavilhão",
            "Água e saneamento mês de janeiro",
            "Fatura bimestral de eletricidade",
        ],
    },
    "Comunicações": {
        "fornecedores": ["NOS", "MEO", "Vodafone", "NOWO", "NOS Empresas", "MEO Empresas"],
        "valor_min": 20, "valor_max": 100,
        "descricoes": [
            "Mensalidade internet e telefone",
            "Fatura telemóvel treinador",
            "Serviço de internet banda larga",
            "Pack comunicações sede do clube",
            "Fatura telefonia fixa mensal",
            "Plano de dados móveis equipa técnica",
        ],
    },
    "Seguros": {
        "fornecedores": ["Fidelidade", "Allianz", "Zurich", "Generali", "Tranquilidade", "AXA", "Ageas"],
        "valor_min": 100, "valor_max": 1500,
        "descricoes": [
            "Seguro de acidentes pessoais atletas",
            "Apólice de seguro responsabilidade civil",
            "Renovação seguro instalações desportivas",
            "Seguro de saúde equipa técnica",
            "Prémio de seguro anual clube",
            "Seguro de equipamentos e material",
        ],
    },
    "Alimentação e Bebidas": {
        "fornecedores": ["Continente", "Pingo Doce", "Intermarché", "Lidl", "Mercadona", "Auchan", "El Corte Inglés"],
        "valor_min": 20, "valor_max": 300,
        "descricoes": [
            "Compras alimentares para evento desportivo",
            "Refeições pós-jogo para atletas",
            "Água e bebidas isotónicas treinos",
            "Catering jantar de gala do clube",
            "Snacks e lanches concentração",
            "Compras supermercado para torneio",
        ],
    },
    "Transportes e Deslocações": {
        "fornecedores": ["Galp", "BP", "Repsol", "CP Comboios", "Uber", "Rentalhas", "Hertz", "FlixBus"],
        "valor_min": 10, "valor_max": 200,
        "descricoes": [
            "Combustível deslocação a jogo fora",
            "Aluguer de autocarro para viagem",
            "Bilhetes de comboio campeonato",
            "Portagens deslocação equipa",
            "Combustível viagem torneio nacional",
            "Transporte atletas para competição",
            "Táxi ou uber após jogo tardio",
        ],
    },
    "Publicidade e Marketing": {
        "fornecedores": ["Publico", "NMedia", "Printmaster", "Gráfica Central", "Design Studio", "Agência Criativa"],
        "valor_min": 50, "valor_max": 500,
        "descricoes": [
            "Produção de flyers para torneio",
            "Campanha nas redes sociais",
            "Impressão de cartazes e banners",
            "Design de logótipo do clube",
            "Publicidade jornal local",
            "Criação de site e presença online",
            "Patrocínio e materiais de merchandising",
        ],
    },
    "Serviços de Contabilidade": {
        "fornecedores": ["PricewaterhouseCoopers", "Deloitte", "BDO", "TOC Independente", "Contabilidade Silva & Associados", "KPMG"],
        "valor_min": 100, "valor_max": 600,
        "descricoes": [
            "Serviços de contabilidade mensal",
            "Declaração fiscal anual do clube",
            "Consultoria financeira gestão associação",
            "Processamento de salários e recibos verdes",
            "Auditoria às contas do exercício",
            "Relatório de contas anual",
        ],
    },
    "Equipamento Informático": {
        "fornecedores": ["Worten", "Fnac", "Apple", "Dell", "HP", "Lenovo", "Samsung", "Microsoft"],
        "valor_min": 50, "valor_max": 2000,
        "descricoes": [
            "Compra de portátil para escritório",
            "Tablet para gestão de treinos",
            "Impressora para secretaria do clube",
            "Software de gestão desportiva",
            "Monitor e periféricos computador",
            "Webcam e microfone reuniões online",
            "Disco externo backup documentos",
        ],
    },
    "Arbitragem e Competições": {
        "fornecedores": ["Federação Portuguesa de Andebol", "Associação Regional de Andebol", "Árbitro Silva", "Árbitro Costa", "Delegado Ferreira"],
        "valor_min": 50, "valor_max": 400,
        "descricoes": [
            "Pagamento de árbitros jogo casa",
            "Inscrição campeonato nacional",
            "Quotas federação portuguesa de andebol",
            "Taxa de participação torneio",
            "Honorários árbitro assistente",
            "Custos organização jogo oficial",
        ],
    },
    "Manutenção e Reparações": {
        "fornecedores": ["Fixando", "ManutençãoPro", "Electricista Costa", "Serralharia Norte", "Pinturas Lisboa", "TecnoFix"],
        "valor_min": 30, "valor_max": 500,
        "descricoes": [
            "Reparação balneários instalações",
            "Manutenção sistema elétrico pavilhão",
            "Substituição de equipamentos danificados",
            "Pintura e remodelação escritório",
            "Arranjo de portões e vedações",
            "Serviço de limpeza e manutenção geral",
            "Reparação canalização balneários",
        ],
    },
}

TOTAL_RECORDS = 600
NOISE_FRACTION = 0.08

all_categories = list(CATEGORIES.keys())
records = []
per_category = TOTAL_RECORDS // len(all_categories)

for cat, info in CATEGORIES.items():
    for _ in range(per_category):
        fornecedor = random.choice(info["fornecedores"])
        descricao = random.choice(info["descricoes"])

        mean_val = (info["valor_min"] + info["valor_max"]) / 2
        std_val = (info["valor_max"] - info["valor_min"]) / 4
        valor = np.random.normal(mean_val, std_val)
        valor = max(info["valor_min"] * 0.8, min(info["valor_max"] * 1.2, valor))
        valor = round(valor, 2)

        records.append({"fornecedor": fornecedor, "valor": valor, "descricao": descricao, "categoria": cat})

remaining = TOTAL_RECORDS - len(records)
for _ in range(remaining):
    cat = random.choice(all_categories)
    info = CATEGORIES[cat]
    fornecedor = random.choice(info["fornecedores"])
    descricao = random.choice(info["descricoes"])
    mean_val = (info["valor_min"] + info["valor_max"]) / 2
    std_val = (info["valor_max"] - info["valor_min"]) / 4
    valor = round(max(info["valor_min"] * 0.8, min(info["valor_max"] * 1.2, np.random.normal(mean_val, std_val))), 2)
    records.append({"fornecedor": fornecedor, "valor": valor, "descricao": descricao, "categoria": cat})

noise_count = int(TOTAL_RECORDS * NOISE_FRACTION)
noise_indices = random.sample(range(TOTAL_RECORDS), noise_count)
for idx in noise_indices:
    original_cat = records[idx]["categoria"]
    other_cats = [c for c in all_categories if c != original_cat]
    noise_cat = random.choice(other_cats)
    noise_info = CATEGORIES[noise_cat]
    records[idx]["fornecedor"] = random.choice(noise_info["fornecedores"])

random.shuffle(records)

df = pd.DataFrame(records)

output_path = os.path.join(os.path.dirname(__file__), "faturas_sinteticas.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"Total de registos gerados: {len(df)}")
print(f"Registos com ruído (fornecedor de outra categoria): ~{noise_count} ({NOISE_FRACTION*100:.0f}%)")
print("\nContagem por categoria:")
counts = df["categoria"].value_counts()
for cat in all_categories:
    print(f"  {cat}: {counts.get(cat, 0)}")
