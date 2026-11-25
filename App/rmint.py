"""Capital interno de risco de modelo: desvio medido por modelo em indicadores internos de performance.

Capital_i = E_i × D_i(α, h) × (1 + MC(nota_i))

E   exposição (saldo contábil em crédito; valor justo absoluto em mercado)
D   desvio do modelo no quantil α em % da exposição, no horizonte h do ciclo de validação do tier
MC  margem de conservadorismo pela nota da validação (0 na nota A, 100% sem validação)

Crédito (behavior de PD), por grupo homogêneo (GH) g: desvio absoluto d = BadRate − PD do GH;
    viés b_g só conta se a BadRate acumulada sair do intervalo de Jeffreys de 95% da PD do GH;
    σ_g = raiz(max(0, var(d) − média de PD(1 − PD)/n)), líquida do ruído binomial, só se o teste de
    sobredispersão (χ² a 95%) indicar variação além do ruído binomial; senão σ_g = 0;
    PD* = PD + max(0, b_g + z_α σ_g √h); D_g = max{K(PD*) − K(PD); (PD* − PD)·LGD};
    D = Σ_g w_g D_g, com w_g a participação do GH no saldo.
Mercado N2, por família de instrumento f: excesso do desvio do IPV sobre a tolerância da fonte de referência,
    e = sinal(d)·max(0, |d| − τ); viés só se o teste t rejeitar média zero; D_f = quantil α de |e|, e ~ N(b_f, σ_f²);
    D = Σ_f w_f D_f, com w_f a participação no valor justo. Horizonte de 1 ano (o IPV é controle mensal).
Mercado N3: sensibilidade s a premissas alternativas razoavelmente possíveis (CPC 46, 93(h)(ii)) e diferença
    a para o modelo alternativo; D = max(s, a) × z_α / z_c, com z_c o quantil implícito da sensibilidade.
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
    """Crédito: ciclo de validação do tier (o erro de calibragem fica sem detecção até a validação seguinte).
    Mercado: 1 ano; o IPV é controle mensal que corrige a marcação, e a sensibilidade do nível 3 é uma faixa de
    premissas plausíveis, não uma volatilidade que cresce no tempo."""
    return R.PRAZO_MESES[m.tier] / 12 if m.risco == "RC" else 1.0


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
def estat_gh(pd: float, contratos, bads) -> dict:
    """Estatísticas de um GH na série: viés pelo teste de Jeffreys e σ líquida do ruído binomial.
    As janelas de doze meses se sobrepõem, então o tamanho efetivo é o número médio de contratos por janela."""
    from scipy.stats import beta as _beta
    n = np.asarray(contratos, float)
    b = np.asarray(bads, float)
    ok = n > 0
    n, b = n[ok], b[ok]
    if len(n) == 0:
        return None
    br = b / n
    d = br - pd
    n_ef = float(n.mean())
    br_ac = float(b.sum() / n.sum())
    k_ef = br_ac * n_ef
    lo, hi = (float(_beta.ppf(q, k_ef + 0.5, n_ef - k_ef + 0.5)) for q in (0.025, 0.975))
    significativo = not (lo <= pd <= hi)
    var_obs = float(d.var(ddof=1)) if len(d) >= 3 else 0.0
    var_bin = float(np.mean(pd * (1 - pd) / n))
    # teste de sobredispersão: (k − 1)·var_obs/var_bin ~ χ²(k − 1) se só houver ruído binomial
    from scipy.stats import chi2 as _chi2
    k = len(d)
    estat = (k - 1) * var_obs / var_bin if var_bin > 0 and k >= 3 else 0.0
    crit = float(_chi2.ppf(0.95, k - 1)) if k >= 3 else float("inf")
    sobredisp = estat > crit
    return {"pd": pd, "obs": len(n), "n_medio": n_ef, "br": br_ac, "ic_lo": lo, "ic_hi": hi,
            "significativo": significativo, "vies": (br_ac - pd) if significativo else 0.0,
            "sigma_obs": math.sqrt(var_obs), "sigma_bin": math.sqrt(var_bin), "chi2": estat, "chi2_crit": crit,
            "sobredispersao": sobredisp,
            "sigma": math.sqrt(max(0.0, var_obs - var_bin)) if sobredisp else 0.0}


def desvio_credito_gh(ghs: list, lgd: float, classe: str, h: float, p: ParamInt) -> dict:
    """ghs: lista de dicts com estat_gh e 'saldo'. Desvio do modelo = média dos D_g ponderada pelo saldo."""
    tot = sum(g["saldo"] for g in ghs) or 1.0
    det, dk_t, dpe_t, d_t = [], 0.0, 0.0, 0.0
    for g in ghs:
        pd = g["pd"]
        pd_s = min(pd + max(0.0, g["vies"] + p.z * g["sigma"] * math.sqrt(h)), 0.999)
        dk = max(0.0, k_classe(pd_s, lgd, classe, p.maturidade) - k_classe(pd, lgd, classe, p.maturidade))
        dpe = max(0.0, (pd_s - pd) * lgd)
        w = g["saldo"] / tot
        dg = max(dk, dpe)
        det.append({**g, "pd_estressada": pd_s, "D_K": dk, "D_PE": dpe, "D": dg, "peso": w,
                    "criterio": "Capital IRB" if dk >= dpe else "Provisão"})
        dk_t += w * dk
        dpe_t += w * dpe
        d_t += w * dg
    crit = {x["criterio"] for x in det if x["D"] > 0}
    return {"D": d_t, "D_K": dk_t, "D_PE": dpe_t, "ghs": det,
            "criterio": crit.pop() if len(crit) == 1 else ("Misto" if crit else "-"),
            "pd_media": sum(x["peso"] * x["pd"] for x in det),
            "pd_estressada_media": sum(x["peso"] * x["pd_estressada"] for x in det)}


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


def estat_familia(desvios, tolerancias) -> dict | None:
    """Família de instrumento no IPV: excesso sobre a tolerância e = sinal(d)·max(0, |d| − τ); viés pelo teste t
    (95%, bilateral) sobre o excesso; σ = desvio-padrão do excesso."""
    from scipy.stats import t as _t
    d = np.asarray(desvios, float)
    tau = np.asarray(tolerancias, float)
    n = len(d)
    if n == 0:
        return None
    e = np.sign(d) * np.maximum(0.0, np.abs(d) - tau)
    media = float(e.mean())
    sd = float(e.std(ddof=1)) if n >= 3 else 0.0
    if n >= 3 and sd > 0:
        estat = media / (sd / math.sqrt(n))
        crit = float(_t.ppf(0.975, n - 1))
    else:
        estat, crit = 0.0, float("inf")
    sig = abs(estat) > crit
    return {"obs": n, "d_medio": float(d.mean()), "d_sd": float(d.std(ddof=1)) if n >= 3 else 0.0,
            "tau": float(tau.mean()), "fora_tol": float(np.mean(np.abs(d) > tau)), "e_medio": media, "t": estat,
            "t_crit": crit, "significativo": sig, "vies": media if sig else 0.0, "sigma": sd}


def desvio_n2_familias(fams: list, h: float, p: ParamInt) -> dict:
    """fams: lista de dicts de estat_familia com 'vj'. D = média dos D_f ponderada pelo valor justo."""
    tot = sum(f["vj"] for f in fams) or 1.0
    det, d_t = [], 0.0
    for f in fams:
        df_ = quantil_abs(f["vies"], f["sigma"] * math.sqrt(h), p.alfa)
        w = f["vj"] / tot
        det.append({**f, "D": df_, "peso": w})
        d_t += w * df_
    return {"D": d_t, "familias": det}


def desvio_n3(sens: float | None, alt: float | None, h: float, p: ParamInt) -> dict:
    base = max(sens or 0.0, alt or 0.0)
    esc = p.z / float(norm.ppf(p.conf_sens))
    return {"base": base, "escala": esc, "D": base * esc,
            "criterio": "Modelo alternativo" if (alt or 0) > (sens or 0) else "Sensibilidade CPC 46"}


def desvio(m: R.Modelo, p: ParamInt) -> dict | None:
    """Desvio do modelo; None se faltar indicador."""
    i, h, fam = m.ind, horizonte(m), familia(m)
    if fam == "RC":
        if not i.get("ghs") or i.get("lgd") is None or i.get("classe") is None:
            return None
        return desvio_credito_gh(i["ghs"], i["lgd"], i["classe"], h, p)
    if fam == "N2":
        if not i.get("fams"):
            return None
        return desvio_n2_familias(i["fams"], h, p)
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
