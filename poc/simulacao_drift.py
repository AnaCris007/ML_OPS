"""
Prova de conceito: Detector de Drift da arquitetura proposta.

Simula 14 semanas de uso de um chatbot de atendimento acadêmico e mostra que:
  1. uma mudança no jeito de perguntar (covariate shift) é detectada pelos testes
     de entrada (KS e PSI sobre embeddings), mas não piora as respostas;
  2. uma mudança de regra (deriva real) não aparece nos testes de entrada, mas é
     detectada pelo ADWIN na taxa de reprovação e confirmada no conjunto de referência;
  3. como a deriva confirmada é factual, a resposta proporcional é atualizar a base
     de conhecimento, sem retreinar o modelo.

Uso:
    pip install -r requirements.txt
    python simulacao_drift.py

Saídas em ./resultados: figura (PNG e SVG), tabela semanal (CSV e Markdown) e log de decisões.
Todos os dados são sintéticos e a semente é fixa, então o resultado é reprodutível.
"""
from __future__ import annotations

import csv
import os
import random
import unicodedata

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from river import drift

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Domínio: intenções, perguntas-modelo e respostas (base de conhecimento)
# ----------------------------------------------------------------------------
INTENTS = {
    "reembolso": [
        "Qual é o prazo para pedir reembolso da mensalidade?",
        "Em quantos dias posso solicitar o reembolso?",
        "Até quando consigo pedir o dinheiro de volta?",
        "Como funciona o prazo de reembolso?",
        "Perdi a aula, ainda dá tempo de pedir reembolso?",
    ],
    "matricula": [
        "Quando abre o período de matrícula?",
        "Como faço a rematrícula do próximo semestre?",
        "Qual a data limite para me matricular?",
        "Onde confirmo minha matrícula nas disciplinas?",
        "Preciso fazer a matrícula pelo portal?",
    ],
    "boleto": [
        "Como emito a segunda via do boleto?",
        "Onde encontro o boleto da mensalidade?",
        "Meu boleto venceu, como gero outro?",
        "Posso pagar a mensalidade com boleto atualizado?",
        "O boleto não chegou no meu e-mail, o que faço?",
    ],
    "secretaria": [
        "Qual o horário de atendimento da secretaria?",
        "A secretaria abre no sábado?",
        "Até que horas a secretaria funciona?",
        "Como falo com a secretaria acadêmica?",
        "Qual o telefone da secretaria?",
    ],
    "trancamento": [
        "Como faço para trancar o curso?",
        "Posso trancar a matrícula no meio do semestre?",
        "Qual o procedimento de trancamento?",
        "Trancar o curso tem algum custo?",
        "Quero pausar o curso por um semestre, como funciona?",
    ],
    "biblioteca": [
        "Quantos livros posso pegar na biblioteca?",
        "Qual o prazo de devolução dos livros?",
        "Como renovo um empréstimo da biblioteca?",
        "A biblioteca abre aos domingos?",
        "Como reservo um livro que está emprestado?",
    ],
}
PREFIXOS = ["", "", "Olá, ", "Bom dia! ", "Por favor, ", "Uma dúvida: "]

# Fato que muda na semana 9 (deriva real): prazo de reembolso de 7 para 30 dias.
RESPOSTA_ATUAL = {i: f"resposta_{i}_v1" for i in INTENTS}
SEMANA_MUDANCA_REGRA = 9


def sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def estilo_informal(t: str) -> str:
    """Novo canal (mensageria/voz): sem acento, abreviações, saudações e vícios de fala."""
    t = sem_acento(t.lower())
    for a, b in [("voce", "vc"), ("quando", "qnd"), ("para ", "pra "), ("por favor", "pfv"),
                 ("qual e ", "qual eh "), ("que ", "q "), ("tambem", "tb"), ("?", "")]:
        t = t.replace(a, b)
    inicio = random.choice(["oi tudo bem? ", "eai ", "entao ", "e... ", "boa tarde gente ", "oii "])
    fim = random.choice([" valeu", " obg", " por favor me ajuda", " ?? ", " rs", ""])
    return inicio + t + fim


def gerar(n: int, informal: float) -> list[tuple[str, str]]:
    msgs = []
    for _ in range(n):
        intent = random.choice(list(INTENTS))
        base = random.choice(PREFIXOS) + random.choice(INTENTS[intent])
        texto = estilo_informal(base) if random.random() < informal else base
        msgs.append((texto, intent))
    return msgs


# ----------------------------------------------------------------------------
# 2. Modelo em produção: classificador de intenção + conhecimento congelado
#    (equivale ao "modelo base + adapter ativo" sem atualização)
# ----------------------------------------------------------------------------
treino = gerar(1200, informal=0.10)  # já havia algumas mensagens informais no treino
vec_modelo = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=5.0)
clf.fit(vec_modelo.fit_transform([t for t, _ in treino]), [i for _, i in treino])
CONHECIMENTO_MODELO = dict(RESPOSTA_ATUAL)  # o que o modelo "sabe"; congelado

# ----------------------------------------------------------------------------
# 3. Detector de Drift (camada 2)
# ----------------------------------------------------------------------------
K_COMP, ALPHA = 16, 0.01


class DetectorEntrada:
    """KS por componente (com correção de Bonferroni) e PSI sobre as mesmas projeções."""

    def __init__(self, referencia: list[str], historico: list[list[str]]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        X = self.vec.fit_transform(referencia)
        self.svd = TruncatedSVD(K_COMP, random_state=SEED).fit(X)
        self.Z_ref = self.svd.transform(X)
        # Faixas do PSI: decis de cada projeção na janela de referência
        self.cortes = [np.quantile(self.Z_ref[:, k], np.linspace(0, 1, 11)[1:-1]) for k in range(K_COMP)]
        self.p_ref = [self._hist(self.Z_ref[:, k], k) for k in range(K_COMP)]
        # Calibração do limite de PSI com dados históricos: semanas passadas sem deriva
        psis = [self._psi(self.svd.transform(self.vec.transform(h))) for h in historico]
        self.limite_psi = float(np.quantile(psis, 0.99))

    def _hist(self, z, k):
        c = np.bincount(np.searchsorted(self.cortes[k], z), minlength=10).astype(float)
        return (c + 0.5) / (c.sum() + 5.0)

    def _psi(self, Z):
        """PSI médio entre as projeções."""
        vals = []
        for k in range(K_COMP):
            p = self._hist(Z[:, k], k)
            vals.append(np.sum((p - self.p_ref[k]) * np.log(p / self.p_ref[k])))
        return float(np.mean(vals))

    def avaliar(self, textos: list[str]):
        Z = self.svd.transform(self.vec.transform(textos))
        pvals = [ks_2samp(self.Z_ref[:, k], Z[:, k]).pvalue for k in range(K_COMP)]
        ks_sig = int(sum(p < ALPHA / K_COMP for p in pvals))  # Bonferroni
        psi = self._psi(Z)
        alerta = ks_sig > 0 and psi > self.limite_psi  # os dois testes precisam concordar
        return ks_sig, psi, alerta


def acerto_referencia(conhecimento: dict, n=120) -> float:
    """Conjunto de referência: perguntas com a resposta correta vigente, revisadas por pessoas."""
    amostra = gerar(n, informal=0.5)
    pred = clf.predict(vec_modelo.transform([t for t, _ in amostra]))
    ok = [p == i and conhecimento[p] == RESPOSTA_ATUAL[i] for p, (_, i) in zip(pred, amostra)]
    return float(np.mean(ok))


# ----------------------------------------------------------------------------
# 4. Simulação de 14 semanas
# ----------------------------------------------------------------------------
N_SEMANA = 500
P_REPROVA_ERRADA, P_REPROVA_CERTA = 0.55, 0.04
LIMITE_REPROVACAO_PP = 0.04  # alerta se a taxa semanal superar a linha de base em 4 p.p.


def fase(s: int) -> str:
    if s <= 4:
        return "estavel"
    if s <= 8:
        return "covariate"
    if s <= 10:
        return "deriva_real"
    return "apos_correcao"


def janela(n, informal):
    return [t for t, _ in gerar(n, informal)]


# Referência e histórico (40 semanas passadas sem deriva) para calibrar o limite do PSI
det = DetectorEntrada(janela(1000, 0.10), [janela(N_SEMANA, 0.10) for _ in range(40)])
adwin = drift.ADWIN(delta=0.002)
linha_base = None
semanas_alerta_resultado = 0
investigacao_aberta = False
log, linhas = [], []

for s in range(1, 15):
    f = fase(s)
    if s == SEMANA_MUDANCA_REGRA:
        RESPOSTA_ATUAL["reembolso"] = "resposta_reembolso_v2"  # 7 -> 30 dias
    informal = {"estavel": 0.10}.get(f, 0.75)
    msgs = gerar(N_SEMANA, informal)
    pred = clf.predict(vec_modelo.transform([t for t, _ in msgs]))

    reprov, adwin_disparou = [], False
    for p, (_, i) in zip(pred, msgs):
        correta = (p == i) and (CONHECIMENTO_MODELO[p] == RESPOSTA_ATUAL[i])
        r = random.random() < (P_REPROVA_CERTA if correta else P_REPROVA_ERRADA)
        reprov.append(r)
        adwin.update(int(r))
        adwin_disparou |= adwin.drift_detected
    taxa = float(np.mean(reprov))
    if s <= 4:
        linha_base = taxa if linha_base is None else (linha_base * (s - 1) + taxa) / s

    ks_sig, psi, alerta_entrada = det.avaliar([t for t, _ in msgs])
    acima_limite = s > 4 and taxa > linha_base + LIMITE_REPROVACAO_PP
    semanas_alerta_resultado = semanas_alerta_resultado + 1 if (acima_limite and adwin_disparou) or (acima_limite and semanas_alerta_resultado) else 0
    acc_ref = acerto_referencia(CONHECIMENTO_MODELO)

    decisao = "Monitorar"
    if alerta_entrada and not acima_limite:
        decisao = "Investigar entrada; sem degradação"
        investigacao_aberta = True
    if semanas_alerta_resultado == 1:
        decisao = "Alerta de resultado; aguardar persistência"
    if semanas_alerta_resultado >= 2:
        # Validação da hipótese de degradação no conjunto de referência
        if acc_ref < 0.90:
            # Erros concentrados em uma intenção => deriva factual => atualizar base, sem retreino
            erros = {}
            for p, (_, i) in zip(pred, msgs):
                if not ((p == i) and (CONHECIMENTO_MODELO[p] == RESPOSTA_ATUAL[i])):
                    erros[i] = erros.get(i, 0) + 1
            principal = max(erros, key=erros.get)
            decisao = f"Drift confirmado ({principal}); atualizar base de conhecimento"
            CONHECIMENTO_MODELO[principal] = RESPOSTA_ATUAL[principal]
            semanas_alerta_resultado = 0
            adwin = drift.ADWIN(delta=0.002)
    if s == 8 and investigacao_aberta:
        # Investigação concluiu que a mudança de estilo é benigna: nova janela vira referência
        det = DetectorEntrada(janela(1000, 0.75), [janela(N_SEMANA, 0.75) for _ in range(40)])
        decisao = "Mudança benigna; referência atualizada"
        investigacao_aberta = False

    linhas.append(dict(semana=s, fase=f, ks_componentes=ks_sig, psi=round(psi, 3),
                       limite_psi=round(det.limite_psi, 3), reprovacao=round(taxa, 3),
                       adwin=adwin_disparou, acerto_referencia=round(acc_ref, 3), decisao=decisao))

# ----------------------------------------------------------------------------
# 5. Saídas: tabela e figura
# ----------------------------------------------------------------------------
with open(os.path.join(OUT, "tabela_semanal.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
    w.writeheader()
    w.writerows(linhas)

with open(os.path.join(OUT, "tabela_semanal.md"), "w", encoding="utf-8") as fh:
    fh.write("| Semana | Fase | KS (componentes) | PSI | Reprovação | ADWIN | Acerto na referência | Decisão |\n")
    fh.write("|---|---|---|---|---|---|---|---|\n")
    nomes = {"estavel": "Estável", "covariate": "Covariate shift", "deriva_real": "Deriva real",
             "apos_correcao": "Após correção"}
    for l in linhas:
        fh.write(f"| {l['semana']} | {nomes[l['fase']]} | {l['ks_componentes']} de {K_COMP} | "
                 f"{l['psi']:.3f} | {l['reprovacao']*100:.1f}% | {'sim' if l['adwin'] else 'não'} | "
                 f"{l['acerto_referencia']*100:.0f}% | {l['decisao']} |\n")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib import font_manager  # noqa: E402

_disp = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": [f for f in ["Inter", "Helvetica", "Arial", "DejaVu Sans"] if f in _disp] or ["sans-serif"],
    "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#94A3B8",
    "axes.labelcolor": "#334155", "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
INK, MUTED, GRID = "#0F172A", "#475569", "#E2E8F0"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
sem = [l["semana"] for l in linhas]
FASES = [(0.5, 4.5, "Estável"), (4.5, 8.5, "Mudança de estilo\n(covariate shift)"),
         (8.5, 10.5, "Mudança de regra\n(deriva real)"), (10.5, 14.5, "Após correção")]

fig, axs = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True, gridspec_kw=dict(hspace=0.45))
for ax in axs:
    for i, (a, b, _) in enumerate(FASES):
        ax.axvspan(a, b, color="#F1F5F9" if i % 2 else "#FFFFFF", zorder=0, lw=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
for a, b, nome in FASES:
    axs[0].text((a + b) / 2, 1.42, nome, transform=axs[0].get_xaxis_transform(), ha="center",
                va="top", fontsize=9, color=MUTED)

# (a) Entrada: PSI com limite calibrado
psi = [l["psi"] for l in linhas]
lim = [l["limite_psi"] for l in linhas]
axs[0].plot(sem, psi, color=BLUE, lw=2, marker="o", ms=5, label="PSI semanal")
axs[0].step([0.5] + sem, [lim[0]] + lim, where="post", color=MUTED, lw=1.2, ls="--",
            label="limite calibrado")
axs[0].set_title("(a) Deriva de entrada: PSI médio sobre projeções dos embeddings", color=INK, pad=8)
axs[0].set_ylabel("PSI")
axs[0].axvline(8.5, color=MUTED, lw=1, ls=":")
axs[0].text(8.6, max(psi) * 0.92, "investigação concluída:\nreferência atualizada", fontsize=8, color=MUTED, va="top")
axs[0].legend(frameon=False, fontsize=8.5, loc="upper right")

# (b) Resultado: taxa de reprovação com detecções do ADWIN
rep = [l["reprovacao"] * 100 for l in linhas]
axs[1].plot(sem, rep, color=ORANGE, lw=2, marker="o", ms=5, label="taxa de reprovação")
axs[1].axhline((linha_base + LIMITE_REPROVACAO_PP) * 100, color=MUTED, lw=1.2, ls="--",
               label="limite (linha de base + 4 p.p.)")
det_sem = [l["semana"] for l in linhas if l["adwin"]]
axs[1].scatter(det_sem, [rep[s - 1] for s in det_sem], s=90, facecolor="none", edgecolor=INK,
               lw=1.5, zorder=5, label="detecção do ADWIN")
axs[1].set_title("(b) Degradação de resultado: reprovação das respostas", color=INK)
axs[1].set_ylabel("reprovação (%)")
axs[1].legend(frameon=False, fontsize=8.5, loc="upper left")

# (c) Validação: acerto no conjunto de referência
acc = [l["acerto_referencia"] * 100 for l in linhas]
axs[2].plot(sem, acc, color=GREEN, lw=2, marker="o", ms=5, label="acerto")
axs[2].axhline(90, color=MUTED, lw=1.2, ls="--", label="mínimo aceitável (90%)")
axs[2].set_title("(c) Validação: acerto no conjunto de referência revisado", color=INK)
axs[2].set_ylabel("acerto (%)")
axs[2].set_ylim(min(acc) - 8, 102)
axs[2].set_xlabel("semana")
axs[2].set_xticks(sem)
axs[2].set_xlim(0.5, 14.5)
axs[2].legend(frameon=False, fontsize=8.5, loc="lower left")
corr = [l["semana"] for l in linhas if "atualizar base" in l["decisao"]]
for c in corr:
    for ax in axs[1:]:
        ax.axvline(c + 0.5, color=GREEN, lw=1, ls=":")
    axs[2].text(c + 0.6, min(acc) - 5, "base de\nconhecimento\natualizada", fontsize=8, color=GREEN, va="bottom")

fig.savefig(os.path.join(OUT, "resultado_poc.png"), dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "resultado_poc.svg"), bbox_inches="tight", facecolor="white")

with open(os.path.join(OUT, "tabela_semanal.md"), encoding="utf-8") as fh:
    print(fh.read())
print(f"Linha de base de reprovação: {linha_base*100:.1f}%  |  limite PSI inicial calibrado")
