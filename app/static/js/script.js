const CAT_COLORS = ["cat-0","cat-1","cat-2","cat-3","cat-4","cat-5"];
const catColorMap = {};
let lastResult = null;

// ── TABS ──
function showTab(name, btn) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("sec-" + name).classList.add("active");
  btn.classList.add("active");
  if (name === "historico") loadHistory();
}

// ── DRAG & DROP ──
const dz = document.getElementById("drop-zone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("dragover"); handlePdfUpload(e.dataTransfer.files[0]); });

// ── UPLOAD PDF ──
async function handlePdfUpload(file) {
  if (!file) return;
  const status = document.getElementById("pdf-status");
  status.textContent = "⏳ A ler fatura...";

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/upload-pdf", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) { status.textContent = "⚠️ " + data.error; return; }

    if (data.fornecedor) document.getElementById("fornecedor").value = data.fornecedor;
    if (data.valor)      document.getElementById("valor").value = data.valor;
    if (data.descricao)  document.getElementById("descricao").value = data.descricao;
    if (data.data)       document.getElementById("data").value = data.data;

    status.textContent = "✅ Dados extraídos com sucesso!";
    
    // Faz aparecer o formulário de preenchimento após os dados serem extraídos
    const formContainer = document.getElementById("container-preenchimento");
    if (formContainer) formContainer.style.display = "block";

  } catch (err) {
    console.error(err);
    status.textContent = "❌ Erro no Javascript: " + err.message;
  }
}

// ── CLASSIFICAR ──
async function classify() {
  const fornecedor = document.getElementById("fornecedor").value.trim();
  const valor      = document.getElementById("valor").value.trim();
  const descricao  = document.getElementById("descricao").value.trim();
  const errEl      = document.getElementById("error-msg");

  errEl.style.display = "none";
  if (!fornecedor || !valor || !descricao) { errEl.textContent = "Preenche os campos."; errEl.style.display = "block"; return; }

  const btn = document.getElementById("classify-btn");
  btn.disabled = true; btn.innerHTML = "⏳ A classificar...";

  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fornecedor, valor, descricao }) });
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; errEl.style.display = "block"; return; }

    lastResult = { fornecedor, valor, descricao, data: document.getElementById("data").value, categoria: data.categoria };
    document.getElementById("result-box").style.display = "block";
    document.getElementById("result-cat").value = data.categoria;
    document.getElementById("btn-save").classList.remove("saved");
    document.getElementById("btn-save").textContent = "💾 Guardar no histórico";
  } catch {
    errEl.textContent = "Erro ao classificar."; errEl.style.display = "block";
  } finally { 
    btn.disabled = false; btn.innerHTML = "<span>⚡</span> Classificar Fatura"; 
  }
}

// ── GUARDAR NO HISTÓRICO ──
async function saveToHistory() {
  if (!lastResult) return;
  const btn = document.getElementById("btn-save");
  lastResult.categoria = document.getElementById("result-cat").value;

  await fetch("/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(lastResult) });
  btn.textContent = "✅ Guardado!"; btn.classList.add("saved");
}

// ── CARREGAR HISTÓRICO ──
async function loadHistory() {
  const res = await fetch("/historico");
  const entries = await res.json();
  const el = document.getElementById("historico-content");

  if (!entries.length) { el.innerHTML = `<div class="empty-msg"><div class="empty-icon">📭</div>Sem faturas.</div>`; return; }

  const rows = [...entries].reverse().map(e => `<tr><td>${e.data||"―"}</td><td><strong>${esc(e.fornecedor)}</strong></td><td>${e.valor ? parseFloat(e.valor).toFixed(2)+" €" : "―"}</td><td>${esc(e.descricao)}</td><td><span class="cat-badge ${getCatColor(e.categoria)}">${esc(e.categoria)}</span></td></tr>`).join("");
  el.innerHTML = `<table><thead><tr><th>Data</th><th>Fornecedor</th><th>Valor</th><th>Descrição</th><th>Categoria</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ── LIMPAR HISTÓRICO ──
async function clearHistory() {
  if (!confirm("Limpar histórico?")) return;
  await fetch("/historico", { method: "DELETE" }); loadHistory();
}

// ── HELPERS ──
function getCatColor(cat) { if (!catColorMap[cat]) catColorMap[cat] = CAT_COLORS[Object.keys(catColorMap).length % CAT_COLORS.length]; return catColorMap[cat]; }
function esc(s) { return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
document.getElementById("data").valueAsDate = new Date();