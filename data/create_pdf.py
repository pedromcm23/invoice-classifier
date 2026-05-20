from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Fatura de Exemplo", ln=True, align='C')
pdf.cell(200, 10, txt="Fornecedor: Decathlon", ln=True)
pdf.cell(200, 10, txt="Data: 15 de março de 2026", ln=True)
pdf.cell(200, 10, txt="Descrição: Compra de bolas de andebol para o escalão júnior", ln=True)
pdf.cell(200, 10, txt="Total a pagar: 150,00 EUR", ln=True)

output_path = os.path.join(os.path.dirname(__file__), "fatura_teste.pdf")
pdf.output(output_path)
print(f"PDF criado em {output_path}")
