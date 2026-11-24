"""Camada de cálculo do app: inventário editável → modelos do motor rmcap → capital, régua, matrizes e racional."""
from __future__ import annotations

import datetime as dt
import io
import math
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import beta

import rmcap as R

NOTAS9 = list("ABCDEFGHI")
SEM_VAL = "Sem validação"
NOME_RISCO = {"RC": "Crédito", "RM": "Mercado"}
COD_RISCO = {"Crédito": "RC", "Mercado": "RM"}

# cores da matriz de notas de referência (fundo, texto)
COR_NOTA = {"A": ("#00B050", "#FFFFFF"), "B": ("#00B050", "#FFFFFF"), "D": ("#00B050", "#FFFFFF"),
            "C": ("#FFFF00", "#000000"), "E": ("#FFFF00", "#000000"), "G": ("#C55A11", "#FFFFFF"),
            "F": ("#FF0000", "#FFFFFF"), "H": ("#FF0000", "#FFFFFF"), "I": ("#C00000", "#FFFFFF"),
            "n/a": ("#7F7F7F", "#FFFFFF")}
NOME_FAIXA = {1: "Verde (A)", 2: "Verde (B, D)", 3: "Amarelo (C, E)", 4: "Laranja (G)", 5: "Vermelho (F, H)",
              6: "Vermelho crítico (I)", 7: "Sem validação ou vencida"}
COR_FAIXA = {1: COR_NOTA["A"], 2: COR_NOTA["B"], 3: COR_NOTA["C"], 4: COR_NOTA["G"], 5: COR_NOTA["F"],
             6: COR_NOTA["I"], 7: COR_NOTA["n/a"]}
# matriz: linhas = quantitativo (Robusto no topo), colunas = qualitativo (Insatisfatório à esquerda)
MATRIZ = [["C", "B", "A"], ["F", "E", "D"], ["I", "H", "G"]]
NIVEIS = list(R.NIVEIS)


# ----------------------------------------------------------------------------- formatação pt-BR
def num(x: float, casas: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = float(x)
    return f"{R._r3(x) if casas == 3 else round(x, casas):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, casas: int = 3) -> str:
    return num(100 * x, casas) + "%"


# ----------------------------------------------------------------------------- inventário
def inventario_exemplo() -> pd.DataFrame:
    linhas = []
    for m in R.INVENTARIO:
        linhas.append({"ID": m.id, "Risco": NOME_RISCO[m.risco], "Carteira ou família": m.carteira, "Modelo": m.nome,
                       "Tier": m.tier, "Nível (mercado)": m.nivel if m.risco == "RM" else None,
                       "Nota": m.nota if m.nota else SEM_VAL,
                       "Última validação": m.data_validacao,
                       "Exposição (R$ mi)": float(m.saldo)})
    return pd.DataFrame(linhas)


def para_modelos(df: pd.DataFrame) -> tuple[list[R.Modelo], list[str]]:
    """Converte a tabela editada em modelos do motor; devolve também avisos de linhas ignoradas ou ajustadas."""
    out, avisos = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        id_ = str(row.get("ID") or "").strip() or f"Linha {i + 1}"
        risco = COD_RISCO.get(str(row.get("Risco") or "").strip())
        tier = str(row.get("Tier") or "").strip().upper()
        try:
            exp = float(row.get("Exposição (R$ mi)"))
        except (TypeError, ValueError):
            exp = float("nan")
        if risco is None or tier not in ("G", "M") or not np.isfinite(exp):
            if any(pd.notna(v) and str(v).strip() for v in row.values):
                avisos.append(f"{id_}: ignorado (risco, tier ou exposição não preenchidos).")
            continue
        nota = str(row.get("Nota") or "").strip().upper()
        nota = nota if nota in NOTAS9 else None
        data = row.get("Última validação")
        if data is not None and not (isinstance(data, float) and math.isnan(data)) and pd.notna(data):
            data = pd.Timestamp(data).date()
        else:
            data = None
        if nota and data is None:
            avisos.append(f"{id_}: nota {nota} sem data da última validação; tratado como sem validação.")
        nivel = None
        if risco == "RM":
            nivel = str(row.get("Nível (mercado)") or "").strip().upper()
            if nivel not in ("N2", "N3"):
                avisos.append(f"{id_}: nível de valor justo não informado; considerado N3 (sem redução).")
                nivel = "N3"
        out.append(R.Modelo(id_, risco, str(row.get("Carteira ou família") or ""), str(row.get("Modelo") or ""),
                            tier, nota, data, exp, nivel))
    return out, avisos


# ----------------------------------------------------------------------------- parâmetros e régua
def parametros(lam_m: float, lam_n2: float) -> R.Parametros:
    return R.Parametros(lam_tier={"G": 1.0, "M": lam_m}, lam_nivel={"N3": 1.0, "N2": lam_n2})


def regua_df(p: R.Parametros) -> pd.DataFrame:
    rg = R.regua(p)
    return pd.DataFrame([{"Faixa": f, "Cor (notas)": NOME_FAIXA[f], "FPR crédito": rg["RC"][f],
                          "FPR mercado": rg["RM"][f]} for f in R.FAIXAS])


def combos(p: R.Parametros) -> dict[str, dict]:
    """Situações de tier e nível com o respectivo fator λ."""
    lm, ln = p.lam_tier["M"], p.lam_nivel["N2"]
    return {"Crédito, tier G": ("RC", 1.0), "Crédito, tier M": ("RC", lm),
            "Mercado, tier G, nível 3": ("RM", 1.0), "Mercado, tier G, nível 2": ("RM", ln),
            "Mercado, tier M, nível 3": ("RM", lm), "Mercado, tier M, nível 2": ("RM", lm * ln)}


def matriz_fpr(p: R.Parametros, risco: str, lam: float) -> dict[str, float]:
    return {n: lam * R.fpr_base(p, risco, n) for n in R.NOTAS}


# ----------------------------------------------------------------------------- cálculo
def calcular(p: R.Parametros, inv: list[R.Modelo], data_base: dt.date) -> dict:
    a = R.addon(p, inv, data_base)
    linhas = []
    for m, l in zip([m for r in R.RISCOS for m in inv if m.risco == r], a["linhas"]):
        st = m.status(data_base)
        situacao = "Vigente" if st != "n/a" else ("Vencida" if m.nota and m.data_validacao else SEM_VAL)
        valida_ate = ((pd.Timestamp(m.data_validacao) + pd.DateOffset(months=R.PRAZO_MESES[m.tier])).date()
                      if m.data_validacao else None)
        linhas.append({"ID": m.id, "Risco": NOME_RISCO[m.risco], "Carteira ou família": m.carteira, "Tier": m.tier,
                       "Nível": m.nivel or "-", "Nota informada": m.nota or "-", "Última validação": m.data_validacao,
                       "Válida até": valida_ate, "Situação": situacao, "Nota efetiva": st,
                       "Faixa": R.FAIXA[st], "Cor": NOME_FAIXA[R.FAIXA[st]],
                       "FPR da faixa": R.fpr_base(p, m.risco, st), "λ tier": p.lam_tier[m.tier],
                       "λ nível": p.lam_nivel[m.nivel] if m.risco == "RM" else 1.0,
                       "FPR de risco de modelo": l["fpr"], "Exposição (R$ mi)": m.saldo,
                       "RWA (R$ mi)": l["rwa"], "Capital (R$ mi)": l["addon"]})
    det = pd.DataFrame(linhas)
    res = []
    for r in R.RISCOS:
        v = a["por_risco"][r]
        res.append({"Risco": NOME_RISCO[r], "Modelos": v["n"], "Sem validação vigente": v["nao_validados"],
                    "Exposição (R$ mi)": v["saldo"], "Cobertura de validação": v["cobertura"],
                    "RWA (R$ mi)": v["rwa"], "FPR médio": v["fpr_medio"], "Capital (R$ mi)": v["addon"],
                    "Parcela do capital sem validação": v["parcela_nao_validados"] / v["addon"] if v["addon"] else 0.0})
    tot_cob = (sum(a["por_risco"][r]["cobertura"] * a["por_risco"][r]["saldo"] for r in R.RISCOS) / a["saldo"]
               if a["saldo"] else 0.0)
    res.append({"Risco": "Total", "Modelos": len(inv), "Sem validação vigente": sum(a["por_risco"][r]["nao_validados"]
                for r in R.RISCOS), "Exposição (R$ mi)": a["saldo"], "Cobertura de validação": tot_cob,
                "RWA (R$ mi)": a["rwa"], "FPR médio": a["fpr_medio"], "Capital (R$ mi)": a["addon"],
                "Parcela do capital sem validação": a["parcela_nao_validados"]})
    return {"bruto": a, "detalhe": det, "resumo": pd.DataFrame(res), "cobertura": tot_cob}


def cenario(inv: list[R.Modelo], nota: str | None, data_base: dt.date, so_pendentes: bool = False) -> list[R.Modelo]:
    out = []
    for m in inv:
        if so_pendentes and m.status(data_base) != "n/a":
            out.append(m)
        else:
            out.append(replace(m, nota=nota, data_validacao=data_base if nota else None))
    return out


def cenarios(p: R.Parametros, inv: list[R.Modelo], data_base: dt.date) -> pd.DataFrame:
    atual = R.addon(p, inv, data_base)["addon"]
    linhas = []
    for n in NOTAS9:
        todos = R.addon(p, cenario(inv, n, data_base), data_base)["addon"]
        pend = R.addon(p, cenario(inv, n, data_base, True), data_base)["addon"]
        linhas.append({"Nota": n, "Todos validados com a nota": todos, "Só os pendentes validados com a nota": pend})
    df = pd.DataFrame(linhas)
    teto = R.addon(p, cenario(inv, None, data_base), data_base)["addon"]
    return df, atual, teto


# ----------------------------------------------------------------------------- racional (relatório)
def racional(p: R.Parametros) -> list[dict]:
    """Seções do relatório com o racional dos FPRs; números calculados pelo motor com os parâmetros vigentes."""
    p0 = R.Parametros()
    pc = replace(p0, abordagem="calibrada")
    P, S = p0.P, p0.S
    kk = {"K(1%)": R.k_irb(0.01), "K(2%)": R.k_irb(0.02)}
    sev = R.severidade_base(p0, "RC")
    fb = {r: R.fator_base(p0, r) for r in R.RISCOS}
    cal = {r: R.fpr_calibrado(pc, r) for r in R.RISCOS}
    rg = R.regua(p0)
    lo, hi = R.intervalo_k(p0)
    ciclo = R.lambda_ciclo(p0)
    salv = R.salvaguarda_ordenacao(p)
    cd = {r: R.cauda(p0, r) for r in R.RISCOS}
    pi = R.PROPORCAO_BASILEIA

    sec = []
    sec.append({"titulo": "1. Formato do cálculo",
                "texto": "O capital de risco de modelo segue o formato da abordagem padronizada de Basileia. Cada modelo "
                         "recebe um **FPR de risco de modelo**, tirado de uma régua por cor da nota de validação, que "
                         "multiplica a exposição da carteira que o modelo atende: saldo contábil em crédito (modelo de "
                         "behavior de PD) e valor justo contábil absoluto em mercado (marcação a mercado de nível 2 e 3, "
                         "incluindo hedge contábil). A soma forma o RWA de risco de modelo, e o capital é 8% desse RWA.",
                "formula": r"\text{RWA}_{RM}=\sum_i \lambda_{tier}\,\lambda_{nível}\,\text{FPR}_r(\text{faixa}_i)\times E_i,"
                           r"\qquad \text{Capital}=8\%\times\text{RWA}_{RM}",
                "formula_txt": "RWA = Σ λ_tier × λ_nível × FPR(faixa) × Exposição;  Capital = 8% × RWA"})
    sec.append({"titulo": "2. Probabilidade de falha material P (Basileia 1996, Tabela 1)",
                "texto": "P é a probabilidade de o modelo sair da zona verde do backtesting (5 ou mais estouros em 250 "
                         "dias) quando a cobertura real é de 99%, 98% ou 97%. Vale para os dois eixos da nota: no "
                         "quantitativo, é a chance de o teste de aderência apontar o modelo; no qualitativo, a chance de "
                         "uma falha de processo se materializar em erro.",
                "formula": r"P(\text{nível})=\Pr[X\ge5],\quad X\sim\text{Binomial}(250;\,1-\text{cobertura})",
                "formula_txt": "P = Pr[X ≥ 5], X ~ Binomial(250; 1 − cobertura)",
                "tabela": pd.DataFrame({"Nível": NIVEIS, "Cobertura": ["99%", "98%", "97%"],
                                        "P calculado": [pct(P[l]) for l in NIVEIS],
                                        "BCBS 1996, Tabela 1": ["10,8%", "56,1%", "87,2%"]})})
    sec.append({"titulo": "3. Magnitude da falha S (Basileia 1996, Tabela 2, e fórmula IRB)",
                "texto": "A magnitude é uma proporção π da severidade máxima de cada risco, com π igual aos acréscimos "
                         "ao multiplicador da Tabela 2 de Basileia (0,40; 0,65; 1,00) divididos pelo máximo. Em mercado, "
                         "o pior caso é a punição máxima do semáforo: acréscimo de 1,00 sobre o multiplicador 3, ou 1/3. "
                         "Em crédito, é o aumento do capital IRB quando a PD verdadeira é o dobro da do modelo (fator de "
                         f"dois observado no caso London Whale), com PD 1%, LGD 45% e M 2,5: K(1%) = {pct(kk['K(1%)'])}, "
                         f"K(2%) = {pct(kk['K(2%)'])}.",
                "formula": r"S_r=\pi\cdot S_{r,máx},\quad S_{RM,máx}=\tfrac{1}{3},\quad "
                           r"S_{RC,máx}=\frac{K(2\,PD)}{K(PD)}-1",
                "formula_txt": "S = π × S_máx;  S_RM,máx = 1/3;  S_RC,máx = K(2·PD)/K(PD) − 1",
                "tabela": pd.DataFrame({"Nível": NIVEIS, "π": [num(pi[l]) for l in NIVEIS],
                                        "S mercado": [pct(S["RM"][l]) for l in NIVEIS],
                                        "S crédito": [pct(S["RC"][l]) for l in NIVEIS]})})
    sec.append({"titulo": "4. Dois canais de erro e harmonização com a matriz de cores",
                "texto": "A severidade de cada nota é a média ponderada de dois canais: o quantitativo (P e S do nível "
                         "quanti) e o qualitativo (P e S do nível quali, com peso ω). O peso é escolhido para que a ordem "
                         "das notas reproduza a matriz de cores de referência: o laranja (G) acima do amarelo (C) exige "
                         f"ω < 1; o laranja abaixo do vermelho (F) exige ω > {num(lo)}. Adota-se o centro do intervalo, "
                         f"**ω = {num(p0.k_quali)}**: uma falha qualitativa pesa 80% de uma quantitativa de mesmo grau. "
                         "O modelo sem validação recebe o pior caso nos dois canais.",
                "formula": r"s_r(\text{nota})=8\%\cdot\frac{P_{quanti}S_{r,quanti}+\omega P_{quali}S_{r,quali}}{1+\omega},"
                           r"\qquad s_r(\text{n/a})=8\%\cdot S_{r,máx}",
                "formula_txt": "s = 8% × [P_quanti·S_quanti + ω·P_quali·S_quali] / (1 + ω);  s(n/a) = 8% × S_máx",
                "tabela": pd.DataFrame([{"Nota": n, "Quanti": R.GRADE[n][0] if n != "n/a" else "-",
                                         "Quali": R.GRADE[n][1] if n != "n/a" else "-",
                                         "Severidade crédito": pct(sev[n]),
                                         "Faixa": NOME_FAIXA[R.FAIXA[n]]}
                                        for n in sorted(sev, key=sev.get)])})
    sec.append({"titulo": "5. Cauda de perda Beta-PERT e FPR calibrado",
                "texto": "A perda por risco de modelo é tratada como variável limitada e assimétrica à direita, com "
                         "mínimo na nota A (risco residual de um modelo correto), moda na nota E (centro da matriz) e "
                         "máximo no modelo sem validação. Cada nota recebe o quantil da cauda correspondente à sua "
                         "posição u, e o FPR calibrado é esse quantil dividido por 8%. Parâmetros da cauda: "
                         f"α = {num(cd['RC']['alpha'])}, β = {num(cd['RC']['beta'])} (iguais nos dois riscos).",
                "formula": r"u=\frac{s(\text{nota})-s(A)}{s(\text{n/a})-s(A)},\quad "
                           r"\text{FPR}^{cal}=\frac{a+(b-a)\,B^{-1}(u;\alpha,\beta)}{8\%}",
                "formula_txt": "u = [s(nota) − s(A)] / [s(n/a) − s(A)];  FPR_cal = [a + (b − a)·Beta⁻¹(u; α, β)] / 8%",
                "tabela": pd.DataFrame([{"Nota": n, "u": num(fb["RC"][n]["u"]),
                                         "FPR calibrado crédito": pct(cal["RC"][n]),
                                         "FPR calibrado mercado": pct(cal["RM"][n])}
                                        for n in sorted(cal["RC"], key=cal["RC"].get)])})
    der = []
    for f in R.FAIXAS:
        nts = R.NOTAS_FAIXA[f]
        mc, mm = max(cal["RC"][n] for n in nts), max(cal["RM"][n] for n in nts)
        der.append({"Faixa": NOME_FAIXA[f], "Máximo calibrado crédito": pct(mc), "Régua crédito": pct(rg["RC"][f]),
                    "Folga crédito (p.p.)": num(100 * (rg["RC"][f] - mc)), "Máximo calibrado mercado": pct(mm),
                    "Régua mercado": pct(rg["RM"][f]), "Folga mercado (p.p.)": num(100 * (rg["RM"][f] - mm))})
    sec.append({"titulo": "6. Da calibração à régua padronizada",
                "texto": "As faixas são as cores da matriz, com a nota A separada das demais verdes (melhor nota) e a nota "
                         "I separada das vermelhas (pior nota validada). Em cada risco, o valor da faixa é o **maior FPR "
                         "calibrado das suas notas, arredondado para cima ao primeiro múltiplo de 0,5 p.p.**; nenhuma "
                         "nota recebe FPR abaixo do calibrado. A mesma regra vale para o modelo de mercado sem validação: "
                         "1/3 de Basileia (33,333%) vira 33,5%.",
                "formula": r"\text{FPR}_r(j)=\max\Big\{0{,}5\%\cdot\Big\lceil\tfrac{\max_{n\in j}\text{FPR}^{cal}_r(n)}"
                           r"{0{,}5\%}\Big\rceil;\ \text{FPR}_r(j-1)+0{,}5\%\Big\}",
                "formula_txt": "FPR(faixa) = máximo calibrado da faixa arredondado para cima a 0,5 p.p. "
                               "(e ao menos 0,5 p.p. acima da faixa anterior)",
                "tabela": pd.DataFrame(der)})
    sec.append({"titulo": "7. Ajustes de tier e de nível de valor justo",
                "texto": f"Tier M e nível 2 multiplicam o FPR por **{num(p.lam_tier['M'])}** e "
                         f"**{num(p.lam_nivel['N2'])}**, respectivamente (padrão: 0,65 = π Satisfatório da Tabela 2 de "
                         "Basileia). No tier M, o valor é confirmado pelo ciclo de validação: a relevância implícita no "
                         "prazo (1 ano ÷ 3 anos) vezes a razão do tempo em que um erro fica sem detecção dá 0,658 para o "
                         "nível satisfatório. O nível 2 tem insumos observáveis e menor incerteza que o nível 3 (CPC 46). "
                         "Salvaguarda: um modelo M sem validação nunca pode pesar menos que qualquer modelo G validado.",
                "formula": r"\lambda_M^{ciclo}=\frac{T_G}{T_M}\cdot\frac{f(T_M)}{f(T_G)},\quad "
                           r"f(T)=1-\frac{1-e^{-\eta T}}{\eta T},\quad \eta=-\ln(1-P)",
                "formula_txt": "λ_ciclo = (T_G/T_M) × f(T_M)/f(T_G);  f(T) = 1 − (1 − e^(−ηT))/(ηT);  η = −ln(1 − P)",
                "tabela": pd.DataFrame([{"Nível quali": l, "Tempo com erro, G (1 ano)": pct(v["f_G"]),
                                         "Tempo com erro, M (3 anos)": pct(v["f_M"]), "λ_M pelo ciclo": num(v["lambda"])}
                                        for l, v in ciclo.items()]),
                "nota": "Salvaguarda com os λ atuais: " + "; ".join(
                    f"{NOME_RISCO[r]}: M sem validação {pct(v['M_sem_validacao'])} vs G nota I {pct(v['G_faixa3'])} "
                    f"({'atendida' if v['ok'] else 'NÃO atendida'})" for r, v in salv.items()) + "."})
    sec.append({"titulo": "8. Escopo do capital, premissa prudencial e RAS de cobertura",
                "texto": "O capital de risco de modelo e a declaração de apetite a riscos (RAS) medem populações "
                         "diferentes. O **capital** cobre a fatia material do inventário: os modelos de behavior de PD "
                         "de cada carteira de crédito e os modelos de marcação a mercado de nível 2 e 3, incluindo hedge "
                         "contábil. A **RAS** mede a cobertura de validação do inventário total, com um percentual "
                         "mínimo definido pela instituição; a tolerância abaixo de 100% recai, na prática, sobre os "
                         "modelos de menor materialidade, que não entram no capital.\n\n"
                         "Como **premissa prudencial**, todos os modelos do escopo do capital devem estar com validação "
                         "ativa e em dia. A metodologia, ainda assim, é geral: se um modelo do escopo estiver sem "
                         "validação ou com validação vencida, ele recebe o FPR de pior caso da régua. Essa faixa funciona "
                         "como penalidade por descumprimento da premissa, e não como custo de uma tolerância aceita. Os "
                         "modelos fora do escopo não recebem capital próprio, por baixa materialidade; ficam cobertos "
                         "pela RAS e pelo plano de validação (premissa assumida).",
                "formula": r"\text{Cobertura}_{RAS}=\frac{\text{validados em dia no escopo}+\text{validados em dia fora do escopo}}"
                           r"{\text{modelos no escopo}+\text{modelos fora do escopo}}\ \ge\ \text{mínimo da RAS}",
                "formula_txt": "Cobertura RAS = (validados em dia no escopo + validados em dia fora do escopo) / "
                               "(modelos no escopo + modelos fora do escopo) ≥ mínimo da RAS"})
    sec.append({"titulo": "9. Referências",
                "texto": "Res. CMN 4.557/2017; Res. BCB 229/2022; Res. CMN 4.966/2021; CPC 46; BCBS (1996), Supervisory "
                         "Framework for the Use of Backtesting; BCBS (2004), Basileia II; BCBS (2005), An Explanatory "
                         "Note on the Basel II IRB Risk Weight Functions; BCBS MAR32; EBA/GL/2017/16; Regulamento (UE) "
                         "2016/101; JPMorgan Chase (2013), Report of the Management Task Force (London Whale); Malcolm "
                         "et al. (1959); Vose (2008), Risk Analysis."})
    return sec


def ras(det: pd.DataFrame, tot_fora: int, val_fora: int, cob_min: float) -> dict:
    """Premissa prudencial do escopo do capital e cobertura de validação do inventário total para a RAS."""
    n_esc = len(det)
    vig = det[det["Situação"] == "Vigente"] if n_esc else det
    fora = det[det["Situação"] != "Vigente"] if n_esc else det
    val_esc = len(vig)
    tot = n_esc + tot_fora
    val = val_esc + val_fora
    cob = val / tot if tot else 0.0
    cap_tot = float(det["Capital (R$ mi)"].sum()) if n_esc else 0.0
    cap_fora = float(fora["Capital (R$ mi)"].sum()) if n_esc else 0.0
    return {"escopo_total": n_esc, "escopo_validos": val_esc, "premissa_ok": val_esc == n_esc,
            "descumprimento": fora, "capital_descumprimento": cap_fora,
            "parcela_capital_descumprimento": cap_fora / cap_tot if cap_tot else 0.0,
            "fora_total": tot_fora, "fora_validos": val_fora, "fora_cobertura": val_fora / tot_fora if tot_fora else 1.0,
            "inv_total": tot, "inv_validos": val, "cobertura": cob, "cobertura_min": cob_min, "ras_ok": cob >= cob_min,
            "folga_modelos": val - math.ceil(cob_min * tot - 1e-9)}


def curva_pert(p: R.Parametros) -> dict:
    out = {}
    for r in R.RISCOS:
        c = R.cauda(p, r)
        x = np.linspace(c["min"], c["max"], 300)
        y = beta.pdf((x - c["min"]) / (c["max"] - c["min"]), c["alpha"], c["beta"]) / (c["max"] - c["min"])
        out[r] = (x, y, c)
    return out


def relatorio_html(p: R.Parametros, resumo: pd.DataFrame, data_base: dt.date, rs: dict | None = None) -> str:
    import markdown
    corpo = []
    for s in racional(p):
        corpo.append(f"<h2>{s['titulo']}</h2>")
        corpo.append(markdown.markdown(s["texto"]))
        if s.get("formula"):
            corpo.append(f'<div class="formula">{s["formula_txt"]}</div>')
        if s.get("tabela") is not None:
            corpo.append(s["tabela"].to_html(index=False, border=0, classes="tab"))
        if s.get("nota"):
            corpo.append(f"<p class='nota'>{s['nota']}</p>")
    rg = regua_df(p)
    tab_rg = rg.assign(**{"FPR crédito": rg["FPR crédito"].map(pct), "FPR mercado": rg["FPR mercado"].map(pct)})
    res = resumo.copy()
    for c in ("Exposição (R$ mi)", "RWA (R$ mi)", "Capital (R$ mi)"):
        res[c] = res[c].map(num)
    for c in ("Cobertura de validação", "FPR médio", "Parcela do capital sem validação"):
        res[c] = res[c].map(pct)
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Racional dos FPRs de risco de modelo</title>
<style>body{{font-family:"Times New Roman",Times,serif;max-width:900px;margin:40px auto;color:#111;line-height:1.45}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px;border-bottom:1px solid #999}}
table.tab{{border-collapse:collapse;margin:10px 0;font-size:13px}} .tab th,.tab td{{border:1px solid #bbb;padding:4px 8px;text-align:right}}
.tab th{{background:#eee}} .formula{{margin:12px 0;padding:8px 12px;background:#f4f4f4;border-left:3px solid #555;font-style:italic}}
.tab td:first-child,.tab th:first-child{{text-align:left}} .nota{{font-size:13px;color:#333}}
.aviso{{border:1px solid #333;padding:6px 10px;font-size:12px}}</style></head><body>
<p class="aviso">Inventário e saldos ilustrativos. Metodologia de referência para capital de risco de modelo.</p>
<h1>Capital para risco de modelo: racional dos FPRs</h1>
<p>Data-base do cálculo: {data_base.strftime('%d/%m/%Y')}.</p>
<h2>Régua padronizada vigente (tier G; mercado nível 3)</h2>{tab_rg.to_html(index=False, border=0, classes='tab')}
<h2>Resultado do inventário</h2>{res.to_html(index=False, border=0, classes='tab')}
{ras_html(rs) if rs else ''}
{''.join(corpo)}</body></html>"""


def excel(det: pd.DataFrame, resumo: pd.DataFrame, rg: pd.DataFrame, inv_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        inv_df.to_excel(w, sheet_name="Inventário", index=False)
        det.to_excel(w, sheet_name="Cálculo por modelo", index=False)
        resumo.to_excel(w, sheet_name="Resumo", index=False)
        rg.to_excel(w, sheet_name="Régua", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(12, min(40, len(str(col[0].value)) + 4))
    return buf.getvalue()


def ras_html(rs: dict) -> str:
    linhas = [("Modelos no escopo do capital", f"{rs['escopo_total']}"),
              ("Com validação vigente no escopo", f"{rs['escopo_validos']}"),
              ("Premissa prudencial (100% do escopo em dia)", "Atendida" if rs["premissa_ok"] else "Não atendida"),
              ("Capital de modelos do escopo fora da premissa (R$ mi)", num(rs["capital_descumprimento"])),
              ("Modelos fora do escopo do capital", f"{rs['fora_total']}"),
              ("Validados em dia fora do escopo", f"{rs['fora_validos']}"),
              ("Cobertura de validação do inventário total", pct(rs["cobertura"])),
              ("Mínimo da RAS", pct(rs["cobertura_min"])),
              ("Situação na RAS", "Dentro do apetite" if rs["ras_ok"] else "Fora do apetite")]
    df = pd.DataFrame(linhas, columns=["Indicador", "Valor"])
    return "<h2>Premissa do escopo e RAS de cobertura</h2>" + df.to_html(index=False, border=0, classes="tab")
