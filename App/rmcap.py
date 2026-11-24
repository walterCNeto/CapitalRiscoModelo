"""
rmcap: capital (add-on de Pilar 2) para risco de modelo por carteira, com régua padronizada de FPR de risco de modelo.

Escopo
------
Crédito: um modelo de behavior (PD) por carteira de crédito. Base = saldo contábil da carteira.
Mercado: modelos de marcação a mercado de instrumentos de nível 2 e 3 da hierarquia de valor justo, incluindo
hedge contábil. Base = valor justo contábil (absoluto) das posições marcadas pelo modelo. Nível 1 fica fora.
Risco operacional não faz parte do escopo. Tier G (validação anual) e M (trienal); validação vencida = n/a.

Fórmula
-------
Calibração (fundamento da régua):
1. Severidade-base em dois canais de erro (réguas de Basileia), com peso k do canal qualitativo:
   s_r(nota) = F × [P(quanti)·S_r(quanti) + k·P(quali)·S_r(quali)] / (1 + k);  s_r(n/a) = F × S_r,máx
   P(nível): BCBS 1996, Tabela 1 (coberturas 99/98/97%); S_r(nível) = π(nível) × S_r,máx, π = 0,40/0,65/1,00 (Tabela 2)
   k = 0,80: centro do intervalo (0,612; 1,000) em que a severidade reproduz a matriz de cores de referência
2. Posição na cauda:  u_r(nota) = [s_r(nota) − s_r(A)] / [s_r(n/a) − s_r(A)]
3. Cauda Beta-PERT do risco r: mínimo s_r(A), moda s_r(E), máximo s_r(n/a)
4. FPR calibrado (tier G, nível 3):  FPR_cal,r(nota) = Q_r(u_r(nota)) / F

Régua padronizada por cor (regra operacional):
5. Faixas: 1 Verde A; 2 Verde B, D; 3 Amarelo {C, E}; 4 Laranja {G}; 5 Vermelho {F, H}; 6 Vermelho crítico {I};
   7 Sem validação ou vencida
6. Em cada risco: maior FPR calibrado da faixa arredondado para cima ao múltiplo de 0,5 p.p., cada faixa ao menos um
   passo acima da anterior. Crédito → 1,5 / 6,5 / 9 / 10 / 12 / 15,5 / 24,5%
7. Mercado → 1,5 / 8,5 / 12 / 13,5 / 16 / 21 / 33,5% (sem validação: 1,00/3 de Basileia arredondado para cima)
8. FPR_RM,i = λ_tier × λ_nível × FPR_faixa;  λ_G = 1, λ_M = 0,65; λ_N3 = 1, λ_N2 = 0,65
   λ_M: π(Satisfatório) de Basileia, verificado pelo ciclo de validação (G 1 ano, M 3 anos):
        λ_ciclo = (T_G/T_M) × f(T_M)/f(T_G), f(T) = 1 − (1 − e^(−λT))/(λT), λ = −ln(1 − P) → 0,658 no nível Satisfatório
   λ_N2: π(Satisfatório), hierarquia de valor justo (CPC 46): nível 2 com insumos observáveis
   Salvaguarda: λ_M × FPR(sem validação) ≥ FPR(vermelho crítico) em cada risco
9. RWA_RM = Σ_i FPR_RM,i × E_i;  E_i = saldo contábil (crédito) ou valor justo contábil absoluto (mercado)
10. Capital (add-on) = k × F × RWA_RM

F = 8%; k = 1. Valores do inventário ilustrativos.
"""
from __future__ import annotations

import argparse
import math
import datetime as dt
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import beta, binom, norm

NIVEIS = ("Robusto", "Satisfatório", "Insatisfatório")
NOTAS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "n/a")
RISCOS = ("RC", "RM")
NOME_RISCO = {"RC": "Crédito", "RM": "Mercado"}
TIERS = ("G", "M")
PRAZO_MESES = {"G": 12, "M": 36}
GRADE = {  # (quanti, quali)
    "A": ("Robusto", "Robusto"), "B": ("Robusto", "Satisfatório"), "C": ("Robusto", "Insatisfatório"),
    "D": ("Satisfatório", "Robusto"), "E": ("Satisfatório", "Satisfatório"), "F": ("Satisfatório", "Insatisfatório"),
    "G": ("Insatisfatório", "Robusto"), "H": ("Insatisfatório", "Satisfatório"), "I": ("Insatisfatório", "Insatisfatório"),
}

# ----------------------------------------------------------------------------- réguas de Basileia
BASEL96_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85, 10: 1.00}   # BCBS (1996), Tabela 2
FRTB_MULT = {4: 1.50, 5: 1.70, 6: 1.76, 7: 1.83, 8: 1.88, 9: 1.92, 10: 2.00}  # MAR32.9
PROPORCAO_BASILEIA = {"Robusto": BASEL96_PLUS[5] / BASEL96_PLUS[10],
                      "Satisfatório": BASEL96_PLUS[7] / BASEL96_PLUS[10], "Insatisfatório": 1.0}


def p_saida_verde(cobertura: float, n_dias: int = 250, limite_verde: int = 4) -> float:
    """P(≥ limite_verde+1 estouros em n_dias) para um VaR cuja cobertura real é `cobertura`."""
    return float(1 - binom.cdf(limite_verde, n_dias, 1 - cobertura))


def s_mercado(estouros: int) -> float:
    return BASEL96_PLUS[estouros] / 3.0


def s_mercado_frtb(estouros: int) -> float:
    return FRTB_MULT[estouros] / FRTB_MULT[4] - 1.0


def k_irb(pd: float, lgd: float = 0.45, m: float = 2.5) -> float:
    """Capital K da fórmula ASRF do IRB para corporate (BCBS, 2005)."""
    w = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
    r = 0.12 * w + 0.24 * (1 - w)
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    ma = (1 + (m - 2.5) * b) / (1 - 1.5 * b)
    return float(lgd * (norm.cdf((norm.ppf(pd) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)) - pd) * ma)


def s_credito(mult: float, pd: float = 0.01, lgd: float = 0.45, m: float = 2.5) -> float:
    return k_irb(min(pd * mult, 0.999), lgd, m) / k_irb(pd, lgd, m) - 1.0


def mult_implicito(s_alvo: float, fn, lo: float = 1.0, hi: float = 10.0) -> float:
    """Erro de insumo m que produz a severidade s_alvo na régua fn (bisseção)."""
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) < s_alvo else (lo, mid)
    return (lo + hi) / 2


# ----------------------------------------------------------------------------- parâmetros
@dataclass
class Calibracao:
    coberturas: dict = field(default_factory=lambda: {"Robusto": 0.99, "Satisfatório": 0.98, "Insatisfatório": 0.97})
    estouros_mercado: dict = field(default_factory=lambda: {"Robusto": 5, "Satisfatório": 7, "Insatisfatório": 10})
    mult_max: float = 2.0          # erro de insumo do pior caso em crédito (fator de dois, caso London Whale)
    pd_ref: float = 0.01
    lgd_ref: float = 0.45


@dataclass
class Parametros:
    P: dict = field(default_factory=dict)
    S: dict = field(default_factory=dict)
    F: float = 0.08
    k: float = 1.0
    moda: str = "E"
    k_quali: float = 0.80          # peso do canal qualitativo relativo ao quantitativo (centro do intervalo da matriz de cores)
    abordagem: str = "padronizada"   # "padronizada" (régua de 5 faixas) ou "calibrada" (FPR exato da cauda)
    lam_tier: dict = field(default_factory=lambda: {"G": 1.0, "M": PROPORCAO_BASILEIA["Satisfatório"]})
    lam_nivel: dict = field(default_factory=lambda: {"N3": 1.0, "N2": PROPORCAO_BASILEIA["Satisfatório"]})
    cal: Calibracao = field(default_factory=Calibracao)

    def __post_init__(self):
        c = self.cal
        if not self.P:
            self.P = {l: p_saida_verde(c.coberturas[l]) for l in NIVEIS}
        S = {r: dict(v) for r, v in self.S.items()}
        S.setdefault("RM", {l: s_mercado(c.estouros_mercado[l]) for l in NIVEIS})
        rc_max = s_credito(c.mult_max, c.pd_ref, c.lgd_ref)
        S.setdefault("RC", {l: PROPORCAO_BASILEIA[l] * rc_max for l in NIVEIS})
        self.S = S


# ----------------------------------------------------------------------------- inventário
@dataclass
class Modelo:
    id: str
    risco: str          # RC ou RM
    carteira: str       # carteira de crédito ou família de instrumentos
    nome: str
    tier: str           # G ou M
    nota: str | None    # A..I; None = sem validação
    data_validacao: dt.date | None
    saldo: float        # saldo contábil (crédito) ou valor justo contábil absoluto (mercado), R$ milhões
    nivel: str | None = None   # N2 ou N3 (mercado)
    observacao: str = ""

    def status(self, data_base: dt.date) -> str:
        """Nota vigente ou n/a (sem validação ou validação vencida pelo prazo do tier)."""
        if self.nota is None or self.data_validacao is None:
            return "n/a"
        m = (data_base.year - self.data_validacao.year) * 12 + data_base.month - self.data_validacao.month
        if data_base.day < self.data_validacao.day:
            m -= 1
        return self.nota if m < PRAZO_MESES[self.tier] else "n/a"


D = dt.date
DATA_BASE = D(2026, 9, 30)
INVENTARIO = [
    # crédito: um modelo de behavior (PD) por carteira
    Modelo("RC-01", "RC", "Veículos", "Behavior PD Veículos", "M", "B", D(2025, 3, 14), 9000.0),
    Modelo("RC-02", "RC", "Consignado Público", "Behavior PD Consignado Público", "G", "A", D(2026, 2, 20), 14000.0),
    Modelo("RC-03", "RC", "Consignado Privado", "Behavior PD Consignado Privado", "M", None, None, 3000.0,
           observacao="Sem validação"),
    Modelo("RC-04", "RC", "Consignado INSS", "Behavior PD Consignado INSS", "G", "B", D(2026, 4, 30), 16000.0),
    Modelo("RC-05", "RC", "Imobiliário", "Behavior PD Imobiliário", "G", "D", D(2026, 6, 10), 12000.0),
    Modelo("RC-06", "RC", "Cartão de Crédito", "Behavior PD Cartão", "G", "G", D(2025, 11, 5), 8000.0),
    Modelo("RC-07", "RC", "Banco PF", "Behavior PD Crédito Pessoal", "M", "C", D(2024, 8, 22), 5000.0),
    Modelo("RC-08", "RC", "Empresas Varejo", "Behavior PD Empresas Varejo", "M", "B", D(2025, 9, 1), 7000.0),
    Modelo("RC-09", "RC", "Middle", "Behavior PD Middle", "G", "A", D(2026, 1, 15), 15000.0),
    Modelo("RC-10", "RC", "Atacado", "Behavior PD Atacado", "G", "B", D(2025, 6, 30), 30000.0,
           observacao="Validação vencida (G: prazo de 1 ano)"),
    # mercado: marcação a mercado de nível 2 e 3, incluindo hedge contábil
    Modelo("RM-01", "RM", "Juros", "Curva de juros pré", "G", "A", D(2026, 3, 31), 6000.0, "N2"),
    Modelo("RM-02", "RM", "Câmbio", "Curva de cupom cambial", "G", "B", D(2026, 5, 15), 3000.0, "N2"),
    Modelo("RM-03", "RM", "Opções", "Superfície de volatilidade e opções", "G", "C", D(2025, 12, 10), 1200.0, "N3"),
    Modelo("RM-04", "RM", "Crédito privado", "Marcação de debêntures e crédito privado", "M", "D", D(2024, 10, 1), 4000.0,
           "N2"),
    Modelo("RM-05", "RM", "Derivativos", "Ajuste de crédito (CVA)", "M", "F", D(2025, 4, 18), 600.0, "N3"),
    Modelo("RM-06", "RM", "Estruturados", "Swaps e operações estruturadas", "G", None, None, 800.0, "N3",
           observacao="Sem validação"),
    Modelo("RM-07", "RM", "Hedge contábil", "Hedge de valor justo: item protegido e efetividade", "G", "B",
           D(2026, 7, 20), 5000.0, "N2"),
    Modelo("RM-08", "RM", "Hedge contábil", "Hedge de fluxo de caixa: efetividade", "M", "C", D(2023, 6, 12), 2500.0,
           "N2", observacao="Validação vencida (M: prazo de 3 anos)"),
]


# ----------------------------------------------------------------------------- núcleo
def severidade_base(p: Parametros, risco: str) -> dict:
    """Dois canais de erro: quantitativo (P e S do nível quanti) e qualitativo (P e S do nível quali, peso k).
    s = F × [P(quanti)·S(quanti) + k·P(quali)·S(quali)] / (1 + k);  sem validação: P = 1 e S máximo nos dois canais."""
    S, k = p.S[risco], p.k_quali
    s = {n: p.F * (p.P[GRADE[n][0]] * S[GRADE[n][0]] + k * p.P[GRADE[n][1]] * S[GRADE[n][1]]) / (1 + k)
         for n in NOTAS[:-1]}
    s["n/a"] = p.F * S["Insatisfatório"]
    return s


def pert_params(minimo: float, moda: float, maximo: float) -> tuple[float, float]:
    h = maximo - minimo
    return 1 + 4 * (moda - minimo) / h, 1 + 4 * (maximo - moda) / h


def cauda(p: Parametros, risco: str) -> dict:
    s = severidade_base(p, risco)
    a, m, b = s["A"], s[p.moda], s["n/a"]
    al, be = pert_params(a, m, b)
    return {"min": a, "moda": m, "max": b, "alpha": al, "beta": be, "media": (a + 4 * m + b) / 6}


def quantil_cauda(c: dict, u: float) -> float:
    return c["min"] + (c["max"] - c["min"]) * float(beta.ppf(u, c["alpha"], c["beta"]))


def fator_base(p: Parametros, risco: str) -> dict:
    """Fator por nota antes dos ajustes de tier e nível (tier G, nível 3): quantil da cauda."""
    s, c = severidade_base(p, risco), cauda(p, risco)
    out = {}
    for n in NOTAS:
        u = 1.0 if n == "n/a" else (s[n] - s["A"]) / (s["n/a"] - s["A"])
        out[n] = {"s": s[n], "u": u, "fator": quantil_cauda(c, u)}
    return out


# ----------------------------------------------------------------------------- régua padronizada
FAIXA = {"A": 1, "B": 2, "D": 2, "C": 3, "E": 3, "G": 4, "F": 5, "H": 5, "I": 6, "n/a": 7}
FAIXAS = (1, 2, 3, 4, 5, 6, 7)
COR_FAIXA = {1: "Verde A", 2: "Verde B, D", 3: "Amarelo", 4: "Laranja", 5: "Vermelho", 6: "Vermelho crítico",
             7: "Sem validação"}
NOTAS_FAIXA = {f: tuple(n for n in NOTAS if FAIXA[n] == f) for f in FAIXAS}
PASSO = 0.005           # arredondamento para cima em múltiplos de 0,5 p.p.


def fpr_calibrado(p: Parametros, risco: str) -> dict:
    """FPR exato por nota (tier G, nível 3): quantil da cauda ÷ F."""
    return {n: v["fator"] / p.F for n, v in fator_base(p, risco).items()}


def regua(p: Parametros) -> dict:
    """Régua padronizada por cor, para cada risco: maior FPR calibrado da faixa arredondado para cima ao múltiplo de
    0,5 p.p., cada faixa ao menos um passo acima da anterior. A mesma regra vale para todas as faixas, inclusive a
    faixa sem validação de mercado (1/3 de Basileia arredondado para 33,5%)."""
    out = {}
    for r in RISCOS:
        ex = fpr_calibrado(p, r)
        v_r, ant = {}, 0.0
        for f in FAIXAS:
            v = math.ceil(max(ex[n] for n in NOTAS_FAIXA[f]) / PASSO - 1e-9) * PASSO
            if f > 1:
                v = max(v, ant + PASSO)
            v_r[f] = ant = v
        out[r] = v_r
    return out


def ordem_cores_ok(p: Parametros) -> bool:
    """A severidade-base respeita a matriz de cores: verde < amarelo < laranja < vermelho < I < sem validação."""
    s = severidade_base(p, "RC")
    return all(max(s[n] for n in NOTAS_FAIXA[f]) < min(s[n] for n in NOTAS_FAIXA[f + 1]) for f in FAIXAS[:-1])


def intervalo_k(p: Parametros, passo: float = 0.0005) -> tuple[float, float]:
    """Intervalo de pesos do canal qualitativo compatível com a matriz de cores."""
    ks = [i * passo for i in range(1, int(2 / passo)) if ordem_cores_ok(replace(p, k_quali=i * passo))]
    return (min(ks), max(ks)) if ks else (float("nan"), float("nan"))


def fpr_base(p: Parametros, risco: str, nota: str) -> float:
    """FPR de risco de modelo antes de λ, pela abordagem escolhida."""
    if p.abordagem == "calibrada":
        return fpr_calibrado(p, risco)[nota]
    return regua(p)[risco][FAIXA[nota]]


def fracao_tempo_com_erro(P: float, anos: float) -> float:
    """Fração do tempo em que o modelo opera com erro não detectado: falhas a taxa λ = −ln(1 − P) ao ano,
    detectadas na validação seguinte (ciclo de `anos`)."""
    if P >= 1.0:
        return 1.0
    lt = -math.log(1.0 - P) * anos
    return 1.0 - (1.0 - math.exp(-lt)) / lt


def lambda_ciclo(p: Parametros) -> dict:
    """Verificação do λ_M pelo ciclo de validação: relevância implícita no prazo (T_G / T_M) × razão do tempo
    com erro não detectado (f(T_M) / f(T_G)), por nível qualitativo."""
    tg, tm = PRAZO_MESES["G"] / 12, PRAZO_MESES["M"] / 12
    out = {}
    for l in NIVEIS:
        fg, fm = fracao_tempo_com_erro(p.P[l], tg), fracao_tempo_com_erro(p.P[l], tm)
        out[l] = {"f_G": fg, "f_M": fm, "razao": fm / fg, "relevancia": tg / tm, "lambda": (tg / tm) * fm / fg}
    return out


def salvaguarda_ordenacao(p: Parametros) -> dict:
    """Modelo sem validação de tier M não pode pesar menos que qualquer modelo validado de tier G (inclusive nota I)."""
    out = {}
    for r in RISCOS:
        m_na = p.lam_tier["M"] * fpr_base(p, r, "n/a")
        g_f3 = max(fpr_base(p, r, n) for n in NOTAS[:-1])
        out[r] = {"M_sem_validacao": m_na, "G_faixa3": g_f3, "ok": m_na >= g_f3 - 1e-12}
    return out


def lam(p: Parametros, m: Modelo) -> float:
    return p.lam_tier[m.tier] * (p.lam_nivel[m.nivel] if m.risco == "RM" else 1.0)


def fator_modelo(p: Parametros, m: Modelo, data_base: dt.date = DATA_BASE) -> float:
    """Fator de capital do modelo (fração do saldo): λ × quantil da cauda."""
    return lam(p, m) * fator_base(p, m.risco)[m.status(data_base)]["fator"]


def fpr_rm(p: Parametros, m: Modelo, data_base: dt.date = DATA_BASE) -> float:
    """FPR de risco de modelo: λ × FPR da faixa (régua padronizada) ou λ × FPR calibrado."""
    return lam(p, m) * fpr_base(p, m.risco, m.status(data_base))


def tabela_fpr(p: Parametros) -> dict:
    """FPR de risco de modelo por risco, nota e combinação de λ (tier G/M; mercado também nível N3/N2)."""
    out = {}
    for r in RISCOS:
        combos = {"G": 1.0, "M": p.lam_tier["M"]} if r == "RC" else {
            "G-N3": 1.0, "G-N2": p.lam_nivel["N2"], "M-N3": p.lam_tier["M"], "M-N2": p.lam_tier["M"] * p.lam_nivel["N2"]}
        out[r] = {k: {n: l * fpr_base(p, r, n) for n in NOTAS} for k, l in combos.items()}
    return out


def addon(p: Parametros, inv: list[Modelo] = INVENTARIO, data_base: dt.date = DATA_BASE) -> dict:
    """RWA de risco de modelo = Σ FPR_RM × saldo; capital = k × F × RWA."""
    linhas, por = [], {}
    for r in RISCOS:
        ms = [m for m in inv if m.risco == r]
        rwa = cap = s_val = 0.0
        for m in ms:
            st = m.status(data_base)
            w = fpr_rm(p, m, data_base)
            rw = w * m.saldo
            k = p.k * p.F * rw
            linhas.append({"id": m.id, "risco": r, "carteira": m.carteira, "nome": m.nome, "tier": m.tier,
                           "nivel": m.nivel, "nota": st, "saldo": m.saldo, "lam": lam(p, m), "fpr": w,
                           "rwa": rw, "addon": k})
            rwa += rw; cap += k; s_val += m.saldo if st != "n/a" else 0.0
        saldo = sum(m.saldo for m in ms)
        por[r] = {"n": len(ms), "nao_validados": sum(m.status(data_base) == "n/a" for m in ms),
                  "saldo": saldo, "cobertura": s_val / saldo if saldo else 0.0, "rwa": rwa, "addon": cap,
                  "fpr_medio": rwa / saldo if saldo else 0.0, "addon_pct_saldo": cap / saldo if saldo else 0.0,
                  "parcela_nao_validados": sum(l["addon"] for l in linhas if l["risco"] == r and l["nota"] == "n/a")}
    total = sum(v["addon"] for v in por.values())
    rwa = sum(v["rwa"] for v in por.values())
    saldo = sum(v["saldo"] for v in por.values())
    return {"linhas": linhas, "por_risco": por, "addon": total, "rwa": rwa, "saldo": saldo,
            "fpr_medio": rwa / saldo if saldo else 0.0, "addon_pct_saldo": total / saldo if saldo else 0.0,
            "parcela_nao_validados": sum(v["parcela_nao_validados"] for v in por.values()) / total if total else 0.0}


def limites(p: Parametros) -> dict:
    """Faixa do FPR de risco de modelo por risco: menor (nota A, menores λ) e maior (n/a, λ = 1)."""
    out = {}
    for r in RISCOS:
        lmin = min(p.lam_tier.values()) * (min(p.lam_nivel.values()) if r == "RM" else 1.0)
        out[r] = {"piso": lmin * min(fpr_base(p, r, n) for n in NOTAS), "teto": fpr_base(p, r, "n/a")}
    return out


def validar_parametros(p: Parametros) -> list[str]:
    e = []
    if not all(0 <= x <= 1 for x in p.P.values()) or not p.P["Robusto"] <= p.P["Satisfatório"] <= p.P["Insatisfatório"]:
        e.append("P deve estar em [0, 1] e crescer de Robusto para Insatisfatório")
    for r in RISCOS:
        s = p.S[r]
        if not all(0 <= x <= 1 for x in s.values()) or not s["Robusto"] <= s["Satisfatório"] <= s["Insatisfatório"]:
            e.append(f"S_{r} deve estar em [0, 1] e crescer de Robusto para Insatisfatório")
    if not 0 < p.F <= 1:
        e.append("F deve estar entre 0% e 100%")
    if p.k < 1:
        e.append("k deve ser ≥ 1")
    if not (0 < p.lam_tier["M"] <= p.lam_tier["G"] == 1.0 and 0 < p.lam_nivel["N2"] <= p.lam_nivel["N3"] == 1.0):
        e.append("λ deve satisfazer 0 < λ_M ≤ λ_G = 1 e 0 < λ_N2 ≤ λ_N3 = 1")
    else:
        if not all(v["ok"] for v in salvaguarda_ordenacao(p).values()):
            e.append("salvaguarda: modelo M sem validação não pode pesar menos que qualquer modelo G validado")
    if p.abordagem not in ("padronizada", "calibrada"):
        e.append("abordagem deve ser padronizada ou calibrada")
    if not ordem_cores_ok(p):
        e.append("a severidade-base não respeita a matriz de cores (ajustar o peso do canal qualitativo k)")
    if p.moda not in ("B", "C", "D", "E", "F", "G", "H"):
        e.append("a moda deve ser uma nota intermediária (entre A e I)")
    return e


def cenario(inv: list[Modelo], nota: str | None, so_pendentes: bool = False) -> list[Modelo]:
    """Inventário com as notas substituídas (nota=None: sem validação). so_pendentes: só os n/a recebem a nota."""
    out = []
    for m in inv:
        if so_pendentes:
            out.append(m if m.status(DATA_BASE) != "n/a" else replace(m, nota=nota, data_validacao=DATA_BASE))
        else:
            out.append(replace(m, nota=nota, data_validacao=DATA_BASE if nota else None))
    return out


def casos_limite(p: Parametros, inv: list[Modelo] = INVENTARIO) -> list[dict]:
    cen = [("Piso: tudo validado, nota A", cenario(inv, "A")),
           ("Pendentes validados com nota B", cenario(inv, "B", True)),
           ("Inventário atual", inv),
           ("Tudo validado, nota I", cenario(inv, "I")),
           ("Teto: nenhum modelo validado", cenario(inv, None))]
    out = []
    for nome, iv in cen:
        a = addon(p, iv)
        out.append({"cenario": nome, "addon": a["addon"], "rwa": a["rwa"], "fpr_medio": a["fpr_medio"],
                    "addon_risco": {r: a["por_risco"][r]["addon"] for r in RISCOS},
                    "fpr_risco": {r: a["por_risco"][r]["fpr_medio"] for r in RISCOS}})
    return out


def notas_descobertas(p_cal: Parametros, r_fixa: dict) -> list[tuple[str, str]]:
    """Notas cujo FPR calibrado (com p_cal) supera o valor da faixa na régua fixada r_fixa."""
    pc = replace(p_cal, abordagem="calibrada")
    out = []
    for r in RISCOS:
        ex = fpr_calibrado(pc, r)
        out += [(r, n) for n in NOTAS if ex[n] > r_fixa[r][FAIXA[n]] + 1e-12]
    return out


def sensibilidade(p: Parametros, inv: list[Modelo] = INVENTARIO) -> list[dict]:
    """Capital calibrado sob premissas alternativas, capital pela régua vigente e cobertura nota a nota.
    Premissas de calibração (moda, m_máx, PD) não mudam a régua vigente; λ muda os dois."""
    r0 = regua(p)
    casos = [("Base", p),
             ("Moda D", replace(p, moda="D")), ("Moda F", replace(p, moda="F")),
             ("k 0,65", replace(p, k_quali=0.65)), ("k 1,00", replace(p, k_quali=1.0)),
             ("m_máx 1,5", replace(p, S={}, cal=replace(p.cal, mult_max=1.5))),
             ("m_máx 2,5", replace(p, S={}, cal=replace(p.cal, mult_max=2.5))),
             ("m_máx 3,0", replace(p, S={}, cal=replace(p.cal, mult_max=3.0))),
             ("PD 0,5%", replace(p, S={}, cal=replace(p.cal, pd_ref=0.005))),
             ("PD 2%", replace(p, S={}, cal=replace(p.cal, pd_ref=0.02))),
             ("PD 3%", replace(p, S={}, cal=replace(p.cal, pd_ref=0.03))),
             ("λ_M 0,40", replace(p, lam_tier={"G": 1.0, "M": 0.40})), ("λ_M 1,00", replace(p, lam_tier={"G": 1.0, "M": 1.0})),
             ("λ_N2 0,40", replace(p, lam_nivel={"N3": 1.0, "N2": 0.40})),
             ("λ_N2 1,00", replace(p, lam_nivel={"N3": 1.0, "N2": 1.0}))]
    out = []
    for nome, q in casos:
        cal = addon(replace(q, abordagem="calibrada"), inv)["addon"]
        qr = q if nome.startswith("λ") or nome == "Base" else p   # régua vigente
        out.append({"cenario": nome, "calibrado": cal, "regua": addon(replace(qr, abordagem="padronizada"), inv)["addon"],
                    "descobertas": notas_descobertas(q, r0)})
    return out


# ----------------------------------------------------------------------------- relatório
def _r3(x: float) -> float:
    """Arredondamento comercial (meio para cima) a 3 casas, imune a resíduos binários."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(round(x, 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
def _pct(x): return f"{_r3(100 * x):,.3f}%".replace(",", "X").replace(".", ",").replace("X", ".")
def _num(x): return f"{_r3(x):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")


def relatorio(p: Parametros, inv: list[Modelo] = INVENTARIO) -> str:
    erros = validar_parametros(p)
    a = addon(p, inv)
    L = ["PARAMETROS",
         "  P (comum):  " + "  ".join(f"{l} {_pct(p.P[l])}" for l in NIVEIS)]
    for r in RISCOS:
        L.append(f"  S_{r}:       " + "  ".join(f"{l} {_pct(p.S[r][l])}" for l in NIVEIS))
    L.append(f"  F = {_pct(p.F)}   k = {_num(p.k)}   moda = nota {p.moda}   m_máx = {_num(p.cal.mult_max)}")
    ki = intervalo_k(p)
    L.append(f"  peso do canal qualitativo k_quali = {_num(p.k_quali)}; intervalo compatível com a matriz de cores: "
             f"({_num(ki[0])}; {_num(ki[1])})")
    L.append(f"  λ tier: G {_num(p.lam_tier['G'])}  M {_num(p.lam_tier['M'])}   "
             f"λ nível (mercado): N3 {_num(p.lam_nivel['N3'])}  N2 {_num(p.lam_nivel['N2'])}")
    pc = replace(p, abordagem="calibrada")
    ex = {r: fpr_calibrado(pc, r) for r in RISCOS}
    rg = regua(p)
    sb = severidade_base(p, "RC")
    L += ["", "SEVERIDADE-BASE DE CRÉDITO POR NOTA (ordem crescente) E COR"]
    L += [f"  {n:<4}{COR_FAIXA[FAIXA[n]]:<18}{_pct(sb[n]):>9}" for n in sorted(sb, key=sb.get)]
    L += ["", "DERIVAÇÃO DA RÉGUA (tier G, nível 3)",
          f"{'Faixa':<6}{'Notas':<10}{'RC máx calib.':>14}{'RC régua':>10}{'RM máx calib.':>14}{'RM régua':>10}{'Folga RM':>10}"]
    for f in FAIXAS:
        mr = max(ex["RC"][n] for n in NOTAS_FAIXA[f]); mm = max(ex["RM"][n] for n in NOTAS_FAIXA[f])
        L.append(f"{f:<6}{','.join(NOTAS_FAIXA[f]):<10}{_pct(mr):>14}{_pct(rg['RC'][f]):>10}{_pct(mm):>14}"
                 f"{_pct(rg['RM'][f]):>10}{_pct(rg['RM'][f] - mm):>10}")
    L += ["", f"{'Nota':<5}{'Faixa':>6}{'RC calib.':>11}{'RC régua':>10}{'RM calib.':>11}{'RM régua':>10}"]
    for n in NOTAS:
        L.append(f"{n:<5}{FAIXA[n]:>6}{_pct(ex['RC'][n]):>11}{_pct(rg['RC'][FAIXA[n]]):>10}"
                 f"{_pct(ex['RM'][n]):>11}{_pct(rg['RM'][FAIXA[n]]):>10}")
    L += ["", "VERIFICAÇÃO DO λ_M PELO CICLO DE VALIDAÇÃO (relevância T_G/T_M × razão do tempo com erro)",
          f"{'Nível quali':<16}{'P':>9}{'f(1 ano)':>10}{'f(3 anos)':>11}{'Razão':>8}{'λ ciclo':>9}"]
    for l, v in lambda_ciclo(p).items():
        L.append(f"{l:<16}{_pct(p.P[l]):>9}{_pct(v['f_G']):>10}{_pct(v['f_M']):>11}{_num(v['razao']):>8}{_num(v['lambda']):>9}")
    L.append(f"  λ_M adotado = {_num(p.lam_tier['M'])} (π Satisfatório, Basileia 1996)")
    L += ["", "SALVAGUARDA DE ORDENAÇÃO (M sem validação ≥ qualquer G validado, inclusive nota I)"]
    for r, v in salvaguarda_ordenacao(p).items():
        L.append(f"  {r}: M sem validação {_pct(v['M_sem_validacao'])}  ≥  G nota I {_pct(v['G_faixa3'])}  "
                 + ("OK" if v["ok"] else "FALHA"))
    t = tabela_fpr(p)
    L += ["", f"ABORDAGEM: {p.abordagem}", "TABELA DE FPR DE RISCO DE MODELO", f"{'Nota':<5}" + "".join(f"{r + ' ' + k:>11}" for r in RISCOS for k in t[r])]
    for n in NOTAS:
        L.append(f"{n:<5}" + "".join(f"{_pct(t[r][k][n]):>11}" for r in RISCOS for k in t[r]))
    L += ["", f"{'ID':<6}{'Carteira':<20}{'T':>2}{'Nív':>4}{'Nota':>5}{'Saldo':>12}{'FPR_RM':>9}{'RWA_RM':>11}{'Capital':>10}"]
    for l in a["linhas"]:
        L.append(f"{l['id']:<6}{l['carteira'][:19]:<20}{l['tier']:>2}{(l['nivel'] or '-'):>4}{l['nota']:>5}"
                 f"{_num(l['saldo']):>12}{_pct(l['fpr']):>9}{_num(l['rwa']):>11}{_num(l['addon']):>10}")
    L += ["", f"{'Risco':<6}{'Saldo':>13}{'Cobert.':>10}{'RWA_RM':>12}{'FPR médio':>10}{'Capital':>10}{'% saldo':>9}"]
    for r, v in a["por_risco"].items():
        L.append(f"{r:<6}{_num(v['saldo']):>13}{_pct(v['cobertura']):>10}{_num(v['rwa']):>12}{_pct(v['fpr_medio']):>10}"
                 f"{_num(v['addon']):>10}{_pct(v['addon_pct_saldo']):>9}")
    L += ["", f"RWA DE RISCO DE MODELO = {_num(a['rwa'])}   FPR médio = {_pct(a['fpr_medio'])}",
          f"CAPITAL (add-on) = {_num(a['addon'])} (R$ milhões) = 8% × RWA; {_pct(a['addon_pct_saldo'])} do saldo",
          f"Parcela dos modelos sem validação vigente: {_pct(a['parcela_nao_validados'])}",
          "", "CASOS-LIMITE (capital, RWA, FPR médio)"]
    L += [f"  {c['cenario']:<34}{_num(c['addon']):>10}{_num(c['rwa']):>12}{_pct(c['fpr_medio']):>10}"
          for c in casos_limite(p, inv)]
    L += ["", "SENSIBILIDADE (capital calibrado | capital pela régua vigente | notas acima da faixa)"]
    L += [f"  {c['cenario']:<12}{_num(c['calibrado']):>10}{_num(c['regua']):>10}   "
          + (", ".join(f"{r}:{n}" for r, n in c["descobertas"]) or "nenhuma") for c in sensibilidade(p, inv)]
    L += ["", "Parametros: " + ("OK" if not erros else "ERRO: " + "; ".join(erros))]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Capital para risco de modelo por carteira (FPR de risco de modelo × saldo contábil)")
    ap.add_argument("--moda", default="E")
    ap.add_argument("--abordagem", default="padronizada", choices=("padronizada", "calibrada"))
    a = ap.parse_args()
    print(relatorio(Parametros(moda=a.moda, abordagem=a.abordagem)))


if __name__ == "__main__":
    main()
