"""Capital interno de risco de modelo: desvio medido por modelo em indicadores internos de performance.

Capital_i = E_i × D_i(α, h) × (1 + MC(nota_i))

E   exposição (saldo contábil em crédito; valor justo absoluto em mercado)
D   desvio do modelo no quantil α em % da exposição, no horizonte h do ciclo de validação do tier
MC  margem de conservadorismo pela nota da validação (0 na nota A, 100% sem validação)

Crédito (behavior de PD): razão r = taxa de default observada / PD prevista, ln r ~ N(μ, σ²);
    m = exp(μ + z_α σ √h); D = max{K(m·PD) − K(PD); (m − 1)·PD·LGD}, com K da fórmula IRB da classe.
Mercado N2: desvio do IPV d (% do valor justo) ~ N(μ, σ² h); D = quantil α de |d|.
Mercado N3: sensibilidade s a premissas alternativas razoavelmente possíveis (CPC 46, 93(h)(ii)) e diferença
    a para o modelo alternativo; D = max(s, a) × z_α / z_c × √h, com z_c o quantil implícito da sensibilidade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import rmcap as R

CLASSES = ("Corporativo", "Varejo imobiliário", "Varejo rotativo", "Varejo outros")
FAMILIAS = {"RC": "Crédito (behavior de PD)", "N2": "Mercado nível 2", "N3": "Mercado nível 3"}


# ----------------------------------------------------------------------------- parâmetros
@dataclass
class ParamInt:
    alfa: float = 0.999            # confiança do desvio em um ano
    conf_sens: float = 0.90        # confiança implícita da sensibilidade divulgada do N3 (alinhada ao AVA)
    omega: float = 0.80            # peso do canal qualitativo na margem de conservadorismo
    maturidade: float = 2.5        # prazo efetivo da fórmula IRB corporativa
    base: R.Parametros = field(default_factory=R.Parametros)   # P e π de Basileia para a margem

    @property
    def z(self) -> float:
        return float(norm.ppf(self.alfa))


def familia(m: R.Modelo) -> str:
    return "RC" if m.risco == "RC" else (m.nivel or "N3")


def horizonte(m: R.Modelo) -> float:
    return R.PRAZO_MESES[m.tier] / 12


# ----------------------------------------------------------------------------- margem de conservadorismo
def perda_relativa(p: ParamInt) -> dict:
    """e(nota) = [P(quanti)·π(quanti) + ω·P(quali)·π(quali)] / (1 + ω), com P e π de Basileia (1996);
    sem validação: e = 1."""
    P, pi, w = p.base.P, R.PROPORCAO_BASILEIA, p.omega
    e = {n: (P[q] * pi[q] + w * P[ql] * pi[ql]) / (1 + w) for n, (q, ql) in R.GRADE.items()}
    e["n/a"] = 1.0
    return e


def margem(p: ParamInt) -> dict:
    """MC(nota) = [e(nota) − e(A)] / [1 − e(A)]: 0 na nota A e 100% sem validação."""
    e = perda_relativa(p)
    return {n: (e[n] - e["A"]) / (1 - e["A"]) for n in R.NOTAS}


def ordem_margem_ok(p: ParamInt) -> bool:
    mc = margem(p)
    return all(max(mc[n] for n in R.NOTAS_FAIXA[f]) < min(mc[n] for n in R.NOTAS_FAIXA[f + 1])
               for f in R.FAIXAS[:-1])


# ----------------------------------------------------------------------------- fórmula IRB por classe
def correlacao(pd: float, classe: str) -> float:
    if classe == "Varejo imobiliário":
        return 0.15
    if classe == "Varejo rotativo":
        return 0.04
    if classe == "Varejo outros":
        w = (1 - math.exp(-35 * pd)) / (1 - math.exp(-35))
        return 0.03 * w + 0.16 * (1 - w)
    w = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def k_classe(pd: float, lgd: float, classe: str, mat: float = 2.5) -> float:
    """Capital K por unidade de exposição na fórmula ASRF (BCBS, 2005); ajuste de prazo só no corporativo."""
    pd = min(max(pd, 0.0003), 0.999)
    r = correlacao(pd, classe)
    k = lgd * (norm.cdf((norm.ppf(pd) + math.sqrt(r) * norm.ppf(0.999)) / math.sqrt(1 - r)) - pd)
    if classe == "Corporativo":
        b = (0.11852 - 0.05478 * math.log(pd)) ** 2
        k *= (1 + (mat - 2.5) * b) / (1 - 1.5 * b)
    return float(k)


# ----------------------------------------------------------------------------- desvios
def desvio_credito(pd: float, lgd: float, classe: str, mu: float, sigma: float, h: float, p: ParamInt) -> dict:
    m = math.exp(mu + p.z * sigma * math.sqrt(h))
    pd_m = min(m * pd, 0.999)
    dk = max(0.0, k_classe(pd_m, lgd, classe, p.maturidade) - k_classe(pd, lgd, classe, p.maturidade))
    dpe = max(0.0, (pd_m - pd) * lgd)
    return {"m": m, "D_K": dk, "D_PE": dpe, "D": max(dk, dpe), "criterio": "Capital IRB" if dk >= dpe else "Provisão"}


def quantil_abs(mu: float, s: float, alfa: float) -> float:
    """q tal que Pr(|X| ≤ q) = α, X ~ N(μ, s²)."""
    if s <= 0:
        return abs(mu)
    f = lambda q: norm.cdf((q - mu) / s) - norm.cdf((-q - mu) / s) - alfa
    return float(brentq(f, 0.0, abs(mu) + 20 * s))


def desvio_n2(mu: float, sigma: float, h: float, p: ParamInt) -> dict:
    return {"D": quantil_abs(mu, sigma * math.sqrt(h), p.alfa)}


def desvio_n3(sens: float | None, alt: float | None, h: float, p: ParamInt) -> dict:
    base = max(sens or 0.0, alt or 0.0)
    esc = p.z / float(norm.ppf(p.conf_sens))
    return {"base": base, "escala": esc, "D": base * esc * math.sqrt(h),
            "criterio": "Modelo alternativo" if (alt or 0) > (sens or 0) else "Sensibilidade CPC 46"}


def desvio(m: R.Modelo, p: ParamInt) -> dict | None:
    """Desvio do modelo; None se faltar indicador."""
    i, h, fam = m.ind, horizonte(m), familia(m)
    if fam == "RC":
        req = ("pd", "lgd", "classe", "mu", "sigma")
        if not all(i.get(k) is not None for k in req):
            return None
        return desvio_credito(i["pd"], i["lgd"], i["classe"], i["mu"], i["sigma"], h, p)
    if fam == "N2":
        if i.get("mu") is None or i.get("sigma") is None:
            return None
        return desvio_n2(i["mu"], i["sigma"], h, p)
    if i.get("sens") is None and i.get("alt") is None:
        return None
    return desvio_n3(i.get("sens"), i.get("alt"), h, p)


# ----------------------------------------------------------------------------- capital
def capital(inv: list, p: ParamInt, data_base) -> dict:
    mc = margem(p)
    linhas = []
    for m in inv:
        d = desvio(m, p)
        linhas.append({"m": m, "d": d})
    # modelo sem indicador: maior desvio medido da família (pior caso observado)
    pior = {}
    for l in linhas:
        if l["d"] is not None:
            f = familia(l["m"])
            pior[f] = max(pior.get(f, 0.0), l["d"]["D"])
    out = []
    for l in linhas:
        m, d = l["m"], l["d"]
        st = m.status(data_base)
        fonte = "Medido"
        if d is None:
            f = familia(m)
            if f in pior:
                d, fonte = {"D": pior[f]}, "Pior da família"
            else:
                d, fonte = {"D": R.fpr_base(R.Parametros(), m.risco, "n/a") * 0.08}, "Régua padronizada (sem dados)"
        cap = m.saldo * d["D"] * (1 + mc[st])
        out.append({"modelo": m, "familia": familia(m), "h": horizonte(m), "nota": st, "mc": mc[st], "fonte": fonte,
                    "desvio": d, "D": d["D"], "capital": cap})
    tot = sum(o["capital"] for o in out)
    return {"linhas": out, "capital": tot, "exposicao": sum(o["modelo"].saldo for o in out),
            "margem": mc}


# ----------------------------------------------------------------------------- estimação a partir de séries
def estat_credito(datas, pd_prev, dr_obs) -> dict:
    """μ e σ de ln(DR/PD); cada observação é a razão da janela de 12 meses, logo já em base anual."""
    pdv, drv = np.asarray(pd_prev, float), np.asarray(dr_obs, float)
    ok = (pdv > 0) & (drv > 0)
    lr = np.log(drv[ok] / pdv[ok])
    n = len(lr)
    return {"n": n, "mu": float(lr.mean()) if n else None, "sigma": float(lr.std(ddof=1)) if n >= 3 else None,
            "pd": float(pdv[-1]) if len(pdv) else None}


def estat_ipv(desvios) -> dict:
    d = np.asarray(desvios, float)
    n = len(d)
    return {"n": n, "mu": float(d.mean()) if n else None, "sigma": float(d.std(ddof=1)) if n >= 3 else None}
