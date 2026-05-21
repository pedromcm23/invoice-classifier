import os
from fpdf import FPDF

os.makedirs('faturas_exemplo', exist_ok=True)
faturas = [
    {'nome': '1_EDP_Eletricidade.pdf', 'fornecedor': 'EDP Comercial', 'data': '10 de maio de 2026', 'descricao': 'Fornecimento de energia elétrica pavilhão', 'valor': '215.40', 'n_fatura': 'FT 2026/001'},
    {'nome': '2_Intersport_Material.pdf', 'fornecedor': 'Intersport', 'data': '12 de maio de 2026', 'descricao': 'Compra de bolas de andebol e redes para baliza', 'valor': '450.00', 'n_fatura': 'INV-8492'},
    {'nome': '3_FlixBus_Deslocacao.pdf', 'fornecedor': 'FlixBus', 'data': '15 de maio de 2026', 'descricao': 'Transporte atletas para competição nacional', 'valor': '120.50', 'n_fatura': 'FLX-99882'},
    {'nome': '4_MEO_Internet.pdf', 'fornecedor': 'MEO', 'data': '20 de maio de 2026', 'descricao': 'Serviço de internet banda larga e telefone do clube', 'valor': '45.99', 'n_fatura': 'A/865019'}
]

class FaturaPDF(FPDF):
    def criar_fatura(self, f):
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f['fornecedor'], ln=True, align='L')
        self.set_font('Arial', '', 12)
        self.cell(0, 8, 'Escola de Andebol - Associacao Desportiva', ln=True, align='L')
        self.cell(0, 8, 'NIF: 501234567', ln=True, align='L')
        self.ln(10)
        self.set_font('Arial', 'B', 12)
        n_fat = f['n_fatura']
        self.cell(0, 8, f'Fatura N: {n_fat}', ln=True)
        self.set_font('Arial', '', 12)
        dt = f['data']
        self.cell(0, 8, f'Data de Emissao: {dt}', ln=True)
        self.ln(10)
        self.set_font('Arial', 'B', 12)
        self.cell(140, 10, 'Descricao', border=1)
        self.cell(50, 10, 'Valor', border=1, align='C', ln=True)
        self.set_font('Arial', '', 12)
        desc = f['descricao']
        val = f['valor']
        self.cell(140, 15, desc, border=1)
        self.cell(50, 15, f'{val} EUR', border=1, align='C', ln=True)
        self.ln(15)
        self.set_font('Arial', 'B', 14)
        self.cell(140, 10, 'Total a pagar:', align='R')
        self.cell(50, 10, f'{val} EUR', align='C', ln=True)
        self.output(os.path.join('faturas_exemplo', f['nome']))

for f in faturas:
    pdf = FaturaPDF()
    pdf.criar_fatura(f)

print("PDFs gerados com sucesso.")
