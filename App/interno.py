"""Camada do app para a abordagem interna: inventário com indicadores, séries de monitoramento, capital e cenários."""
from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd

import calculo as C
import rmcap as R
import rmint as I

_DIR = os.path.dirname(os.path.abspath(__file__))
SERIE_CRED = os.path.join(_DIR, "dados", "serie_credito_exemplo.csv")
SERIE_IPV = os.path.join(_DIR, "dados", "serie_ipv_exemplo.csv")

COL_CLASSE, COL_PD, COL_LGD = "Classe IRB", "PD média (%)", "LGD (%)"
COL_MU_C, COL_SG_C = "μ log(DR/PD)", "σ log(DR/PD)"
COL_MU_I, COL_SG_I = "μ desvio IPV (%)", "σ desvio IPV (p.p.)"
COL_SENS, COL_ALT = "Sensibilidade CPC 46 (%)", "Dif. modelo alternativo (%)"
COLS_IND = [COL_CLASSE, COL_PD, COL_LGD, COL_MU_C, COL_SG_C, COL_MU_I, COL_SG_I, COL_SENS, COL_ALT]

EXEMPLO_CRED = {"RC-01": ("Varejo outros", 40.0), "RC-02": ("Varejo outros", 35.0), "RC-03": ("Varejo outros", 45.0),
                "RC-04": ("Varejo outros", 35.0), "RC-05": ("Varejo imobiliário", 20.0),
                "RC-06": ("Varejo rotativo", 75.0), "RC-07": ("Varejo outros", 60.0),
                "RC-08": ("Varejo outros", 45.0), "RC-09": ("Corporativo", 45.0), "RC-10": ("Corporativo", 45.0)}
EXEMPLO_N3 = {"RM-03": (2.0, 2.5), "RM-05": (4.0, None), "RM-06": (3.0, None)}


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- séries
def _ler(arq) -> pd.DataFrame:
    nome = getattr(arq, "name", str(arq)).lower()
    if nome.endswith((".xlsx", ".xls")):
        return pd.read_excel(arq)
    return pd.read_csv(arq, sep=None, engine="python")


def _padroniza(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    cols = {str(c).lower().strip(): c for c in df.columns}
    ren = {}
    for alvo, chaves in mapa.items():
        for k in chaves:
            if k in cols:
                ren[cols[k]] = alvo
                break
    df = df.rename(columns=ren)
    falt = [c for c in mapa if c not in df.columns]
    if falt:
        raise ValueError("Colunas ausentes: " + ", ".join(falt))
    df = df[list(mapa)].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    iso = df["Data"].astype(str).str.match(r"^\d{4}-").all()
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=not iso, errors="coerce")
    for c in mapa:
        if c not in ("ID", "Data"):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df.dropna().sort_values(["ID", "Data"])


def ler_serie_credito(arq) -> pd.DataFrame:
    return _padroniza(_ler(arq), {"ID": ("id", "modelo"), "Data": ("data", "date"),
                                  "PD prevista (%)": ("pd prevista (%)", "pd prevista", "pd"),
                                  "Taxa de default (%)": ("taxa de default (%)", "taxa de default", "dr",
                                                          "default observado")})


def ler_serie_ipv(arq) -> pd.DataFrame:
    return _padroniza(_ler(arq), {"ID": ("id", "modelo"), "Data": ("data", "date"),
                                  "Desvio IPV (%)": ("desvio ipv (%)", "desvio ipv", "desvio", "ipv")})


def estimar_credito(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for id_, g in df.groupby("ID", sort=False):
        e = I.estat_credito(g["Data"], g["PD prevista (%)"] / 100, g["Taxa de default (%)"] / 100)
        out.append({"ID": id_, "Observações": e["n"], "Início": g["Data"].iloc[0].date(), "Fim": g["Data"].iloc[-1].date(),
                    COL_PD: 100 * e["pd"] if e["pd"] is not None else None, COL_MU_C: e["mu"], COL_SG_C: e["sigma"],
                    "Razão DR/PD média": float((g["Taxa de default (%)"] / g["PD prevista (%)"]).mean())})
    return pd.DataFrame(out)


def estimar_ipv(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for id_, g in df.groupby("ID", sort=False):
        e = I.estat_ipv(g["Desvio IPV (%)"])
        out.append({"ID": id_, "Observações": e["n"], "Início": g["Data"].iloc[0].date(), "Fim": g["Data"].iloc[-1].date(),
                    COL_MU_I: e["mu"], COL_SG_I: e["sigma"]})
    return pd.DataFrame(out)


def aplicar(inv_df: pd.DataFrame, est: pd.DataFrame, cols: list[str], risco: str) -> tuple[pd.DataFrame, list[str]]:
    out = inv_df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    feitos, fora = [], []
    for _, r in est.iterrows():
        msk = (out["ID"].astype(str).str.strip() == r["ID"]) & (out["Risco"] == risco)
        if not msk.any():
            fora.append(r["ID"])
            continue
        for c in cols:
            if pd.notna(r.get(c)):
                out.loc[msk, c] = r[c]
        feitos.append(r["ID"])
    msgs = [f"Atualizados: {', '.join(feitos)}." if feitos else "Nenhum modelo do inventário encontrado na série."]
    if fora:
        msgs.append(f"IDs da série fora do inventário ({risco.lower()}): {', '.join(fora)}.")
    return out, msgs


# ----------------------------------------------------------------------------- inventário
def inventario_exemplo() -> pd.DataFrame:
    linhas = []
    for m in R.INVENTARIO:
        linhas.append({"ID": m.id, "Risco": C.NOME_RISCO[m.risco], "Carteira ou família": m.carteira, "Modelo": m.nome,
                       "Tier": m.tier, "Nível (mercado)": m.nivel if m.risco == "RM" else None,
                       "Nota": m.nota if m.nota else C.SEM_VAL, "Última validação": m.data_validacao,
                       "Exposição (R$ mi)": float(m.saldo)})
    df = pd.DataFrame(linhas)
    df[COL_CLASSE] = [EXEMPLO_CRED.get(i, (None, None))[0] for i in df["ID"]]
    for c in COLS_IND[1:]:
        df[c] = np.nan
    df[COL_LGD] = [EXEMPLO_CRED.get(i, (None, np.nan))[1] for i in df["ID"]]
    df[COL_SENS] = [EXEMPLO_N3.get(i, (np.nan, None))[0] for i in df["ID"]]
    df[COL_ALT] = [EXEMPLO_N3.get(i, (None, np.nan))[1] if EXEMPLO_N3.get(i, (None, None))[1] is not None else np.nan
                   for i in df["ID"]]
    df, _ = aplicar(df, estimar_credito(ler_serie_credito(SERIE_CRED)), [COL_PD, COL_MU_C, COL_SG_C], "Crédito")
    df, _ = aplicar(df, estimar_ipv(ler_serie_ipv(SERIE_IPV)), [COL_MU_I, COL_SG_I], "Mercado")
    return df


def para_modelos(df: pd.DataFrame) -> tuple[list[R.Modelo], list[str]]:
    base, avisos = C.para_modelos(df)
    por_id = {str(r.get("ID") or "").strip(): r for _, r in df.iterrows()}
    out = []
    for m in base:
        r = por_id.get(m.id, {})
        ind = {}
        if m.risco == "RC":
            cl = str(r.get(COL_CLASSE) or "").strip()
            ind = {"classe": cl if cl in I.CLASSES else None,
                   "pd": (_f(r.get(COL_PD)) or 0) / 100 or None, "lgd": (_f(r.get(COL_LGD)) or 0) / 100 or None,
                   "mu": _f(r.get(COL_MU_C)), "sigma": _f(r.get(COL_SG_C))}
            if cl and cl not in I.CLASSES:
                avisos.append(f"{m.id}: classe IRB '{cl}' inválida.")
        elif m.nivel == "N2":
            mu, sg = _f(r.get(COL_MU_I)), _f(r.get(COL_SG_I))
            ind = {"mu": mu / 100 if mu is not None else None, "sigma": sg / 100 if sg is not None else None}
        else:
            s, a = _f(r.get(COL_SENS)), _f(r.get(COL_ALT))
            ind = {"sens": s / 100 if s is not None else None, "alt": a / 100 if a is not None else None}
        out.append(replace(m, ind=ind))
    return out, avisos


# ----------------------------------------------------------------------------- cálculo
def calcular(p: I.ParamInt, inv: list[R.Modelo], data_base: dt.date) -> dict:
    a = I.capital(inv, p, data_base)
    linhas = []
    for o in a["linhas"]:
        m, d = o["modelo"], o["desvio"]
        st = o["nota"]
        situacao = "Vigente" if st != "n/a" else ("Vencida" if m.nota and m.data_validacao else C.SEM_VAL)
        valida_ate = ((pd.Timestamp(m.data_validacao) + pd.DateOffset(months=R.PRAZO_MESES[m.tier])).date()
                      if m.data_validacao else None)
        linhas.append({"ID": m.id, "Risco": C.NOME_RISCO[m.risco], "Família": I.FAMILIAS[o["familia"]],
                       "Carteira ou família": m.carteira, "Tier": m.tier, "Nível": m.nivel or "-",
                       "Nota informada": m.nota or "-", "Última validação": m.data_validacao, "Válida até": valida_ate,
                       "Situação": situacao, "Nota efetiva": st, "Horizonte (anos)": o["h"],
                       "Fonte do desvio": o["fonte"], "Fator m (crédito)": d.get("m"),
                       "Desvio capital IRB": d.get("D_K"), "Desvio provisão": d.get("D_PE"),
                       "Critério": d.get("criterio", "-"), "Desvio D": o["D"], "Margem MC": o["mc"],
                       "Taxa de capital": o["D"] * (1 + o["mc"]), "Exposição (R$ mi)": m.saldo,
                       "Capital (R$ mi)": o["capital"]})
    det = pd.DataFrame(linhas)
    res = []
    for fam, nome in I.FAMILIAS.items():
        sub = det[det["Família"] == nome] if len(det) else det
        if not len(sub):
            continue
        cap, exp_ = sub["Capital (R$ mi)"].sum(), sub["Exposição (R$ mi)"].sum()
        nv = sub[sub["Situação"] != "Vigente"]
        res.append({"Família": nome, "Modelos": len(sub), "Sem validação vigente": len(nv), "Exposição (R$ mi)": exp_,
                    "Capital (R$ mi)": cap, "Taxa de capital": cap / exp_ if exp_ else 0.0,
                    "Capital sem validação vigente": nv["Capital (R$ mi)"].sum()})
    cap_t = float(det["Capital (R$ mi)"].sum()) if len(det) else 0.0
    exp_t = float(det["Exposição (R$ mi)"].sum()) if len(det) else 0.0
    nv = det[det["Situação"] != "Vigente"] if len(det) else det
    res.append({"Família": "Total", "Modelos": len(det), "Sem validação vigente": len(nv), "Exposição (R$ mi)": exp_t,
                "Capital (R$ mi)": cap_t, "Taxa de capital": cap_t / exp_t if exp_t else 0.0,
                "Capital sem validação vigente": float(nv["Capital (R$ mi)"].sum()) if len(nv) else 0.0})
    return {"detalhe": det, "resumo": pd.DataFrame(res), "capital": cap_t, "exposicao": exp_t, "margem": a["margem"]}


def referencia_regua(inv: list[R.Modelo], data_base: dt.date) -> float:
    """Capital pela régua padronizada anterior (λ = 0,65), só para comparação."""
    return R.addon(C.parametros(0.65, 0.65), inv, data_base)["addon"]


def cenarios(p: I.ParamInt, inv: list[R.Modelo], data_base: dt.date):
    atual = calcular(p, inv, data_base)["capital"]
    linhas = []
    for n in C.NOTAS9:
        todos = calcular(p, C.cenario(inv, n, data_base), data_base)["capital"]
        pend = calcular(p, C.cenario(inv, n, data_base, True), data_base)["capital"]
        linhas.append({"Nota": n, "Todos validados com a nota": todos, "Só os pendentes validados com a nota": pend})
    teto = calcular(p, C.cenario(inv, None, data_base), data_base)["capital"]
    return pd.DataFrame(linhas), atual, teto
