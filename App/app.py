"""App Streamlit: capital de risco de modelo pela régua padronizada de FPR, com inventário editável."""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calculo as C
import memorial as MEM
import rmcap as R

st.set_page_config(page_title="Capital de Risco de Modelo", layout="wide")

st.markdown("""<style>
.metric-box{border:1px solid #d0d4e4;border-radius:6px;padding:10px 14px;background:#f7f8fc;height:112px;
display:flex;flex-direction:column;justify-content:space-between;box-sizing:border-box}
.metric-box .l{font-size:13px;color:#555;line-height:1.3;min-height:2.6em;overflow:hidden}
.metric-box .v{font-size:24px;font-weight:700;color:#101448;line-height:1.1;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.mx{border-collapse:separate;border-spacing:5px;margin:auto}
.mx td.c{width:86px;height:62px;text-align:center;border-radius:6px;border:1px solid #444;font-family:Arial}
.mx td.c b{font-size:20px;display:block}.mx td.c span{font-size:13px}
.mx td.lab{font-size:11px;color:#444;text-align:right;padding-right:4px}
.mx th{font-size:11px;color:#444;font-weight:normal}
.na{margin:4px auto 0;width:276px;text-align:center;border-radius:6px;background:#7F7F7F;color:#fff;padding:5px;
font-family:Arial;font-size:13px}
.mt{text-align:center;font-weight:700;color:#101448;margin-bottom:2px}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### Parâmetros")
    data_base = st.date_input("Data-base", value=R.DATA_BASE, format="DD/MM/YYYY")
    lam_m = st.number_input("λ tier M", 0.0, 1.0, 0.65, 0.05, format="%.3f",
                            help="Fator do tier M (tier G = 1). Padrão: 0,65 = π Satisfatório de Basileia.")
    lam_n2 = st.number_input("λ nível 2 (mercado)", 0.0, 1.0, 0.65, 0.05, format="%.3f",
                             help="Fator de modelos de nível 2 de valor justo (nível 3 = 1).")
    st.caption("Validade da validação: 12 meses (tier G) e 36 meses (tier M). Validação vencida ou ausente "
               "recebe o FPR de sem validação. Capital = 8% × RWA.")
    st.markdown("### Probabilidade de falha em crédito")
    fonte_p = st.radio("Fonte do P de crédito", ["Volatilidade do Gini", "Basileia 1996"], horizontal=True,
                       help="Gini: P(nível) = Φ(−(Gini de referência do nível − Gini mínimo) / σ), chance de o Gini "
                            "ficar abaixo do mínimo em um ano. Mercado usa sempre Basileia 1996.")
    usar_gini = fonte_p == "Volatilidade do Gini"
    with st.expander("Referências de Gini e volatilidade", expanded=False):
        g_min = st.number_input("Gini mínimo (reprovação, %)", 0.0, 100.0, 100 * R.GINI_PADRAO["minimo"], 1.0,
                                format="%.3f")
        g_r = st.number_input("Gini de referência: Robusto (%)", 0.0, 100.0, 100 * R.GINI_PADRAO["Robusto"], 1.0,
                              format="%.3f")
        g_s = st.number_input("Gini de referência: Satisfatório (%)", 0.0, 100.0,
                              100 * R.GINI_PADRAO["Satisfatório"], 1.0, format="%.3f")
        g_i = st.number_input("Gini de referência: Insatisfatório (%)", 0.0, 100.0,
                              100 * R.GINI_PADRAO["Insatisfatório"], 1.0, format="%.3f")
        sig_man = st.number_input("σ do Gini manual (p.p./ano; 0 = agregada do inventário)", 0.0, 50.0, 0.0, 0.5,
                                  format="%.3f")
    ref_gini = {"minimo": g_min / 100, "Robusto": g_r / 100, "Satisfatório": g_s / 100,
                "Insatisfatório": g_i / 100}
    ref_ok = ref_gini["Robusto"] > ref_gini["Satisfatório"] > ref_gini["Insatisfatório"]
    if usar_gini and not ref_ok:
        st.warning("As referências de Gini precisam decrescer de Robusto para Insatisfatório; usando Basileia 1996.")
        usar_gini = False
    st.markdown("### Apetite a risco (RAS)")
    cob_min = st.number_input("Cobertura mínima de validação do inventário (%)", 0.0, 100.0, 80.0, 1.0,
                              format="%.3f", help="Percentual mínimo de modelos validados e em dia no inventário "
                                                  "total, definido na RAS.") / 100
    tot_fora = st.number_input("Modelos fora do escopo do capital", 0, 10000, 60, 1,
                               help="Demais modelos do inventário, que não entram no cálculo do capital.")
    val_fora = st.number_input("Desses, validados e em dia", 0, 10000, 50, 1)
    if val_fora > tot_fora:
        st.warning("Validados fora do escopo maior que o total; ajustado ao total.")
        val_fora = tot_fora
    if st.button("Restaurar inventário de exemplo"):
        st.session_state.pop("inv", None)
        st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
        st.rerun()

p_base = C.parametros(lam_m, lam_n2)

if "inv" not in st.session_state:
    st.session_state["inv"] = C.inventario_exemplo()

st.title("Capital de risco de modelo")
st.caption("Pilar 2 · FPR de risco de modelo da régua padronizada × exposição = RWA; capital = 8% × RWA. "
           "Valores em R$ milhões. Inventário e saldos ilustrativos.")

aba_inv, aba_res, aba_ras, aba_gini, aba_mat, aba_cen, aba_rac, aba_mem = st.tabs(
    ["Inventário", "Resultado", "Premissa e RAS", "Gini e P de crédito", "Régua e matrizes de FPR", "Cenários",
     "Racional dos FPRs", "Memorial de cálculo"])

# ----------------------------------------------------------------------------- inventário
with aba_inv:
    st.markdown("Edite notas, tiers, níveis, datas da última validação e exposições. Inclua ou exclua linhas na "
                "própria tabela. Para simular um modelo validado, informe a nota **e** uma data dentro da validade.")
    edit = st.data_editor(
        st.session_state["inv"], key=f"editor_{st.session_state.get('editor_ver', 0)}", num_rows="dynamic",
        width="stretch", hide_index=True, height=38 * (len(st.session_state["inv"]) + 3),
        column_config={
            "Risco": st.column_config.SelectboxColumn(options=["Crédito", "Mercado"], required=True),
            "Tier": st.column_config.SelectboxColumn(options=["G", "M"], required=True),
            "Nível (mercado)": st.column_config.SelectboxColumn(options=["N2", "N3"],
                                                                help="Só para mercado; em branco para crédito."),
            "Nota": st.column_config.SelectboxColumn(options=C.NOTAS9 + [C.SEM_VAL], required=True),
            "Última validação": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Exposição (R$ mi)": st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                               help="Crédito: saldo contábil. Mercado: valor justo "
                                                                    "contábil absoluto."),
            "Modelo": st.column_config.TextColumn(width="medium"),
            C.COL_GINI: st.column_config.NumberColumn(format="%.3f", min_value=0.0, max_value=100.0,
                                                      help="Só crédito: Gini da última observação do monitoramento."),
            C.COL_SIGMA: st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                       help="Só crédito: volatilidade anual do Gini na série do "
                                                            "monitoramento. Pode ser preenchida pela aba Gini e P "
                                                            "de crédito."),
        })

inv, avisos = C.para_modelos(edit)
p, info_gini = C.calibrar_gini(p_base, inv, usar_gini, ref_gini, sig_man / 100 if sig_man else None)
if usar_gini and not info_gini["usar"]:
    st.sidebar.warning("Nenhum modelo de crédito com σ do Gini informada; usando Basileia 1996.")
elif info_gini["usar"] and not info_gini["ordem_ok"]:
    st.sidebar.warning("Com estas referências de Gini não há peso ω que reproduza a matriz de cores em crédito. "
                       "Revise as referências ou a σ.")
for r, v in R.salvaguarda_ordenacao(p).items():
    if not v["ok"]:
        st.sidebar.warning(f"Salvaguarda não atendida em {C.NOME_RISCO[r].lower()}: modelo M sem validação "
                           f"({C.pct(v['M_sem_validacao'])}) pesa menos que um G nota I ({C.pct(v['G_faixa3'])}).")
out = C.calcular(p, inv, data_base)
det, resumo, bruto = out["detalhe"], out["resumo"], out["bruto"]
with aba_inv:
    for a in avisos:
        st.warning(a)

# ----------------------------------------------------------------------------- indicadores
def datas(df: pd.DataFrame) -> pd.DataFrame:
    """Datas como texto dd/mm/aaaa e traço quando ausentes, para exibição."""
    out = df.copy()
    for c in ("Última validação", "Válida até"):
        if c in out:
            out[c] = [x.strftime("%d/%m/%Y") if x is not None and pd.notna(x) else "-" for x in out[c]]
    return out


def caixa(col, rotulo, valor):
    col.markdown(f"<div class='metric-box'><div class='l'>{rotulo}</div><div class='v'>{valor}</div></div>",
                 unsafe_allow_html=True)


rs = C.ras(det, int(tot_fora), int(val_fora), cob_min)

# ----------------------------------------------------------------------------- resultado
with aba_res:
    if not rs["premissa_ok"]:
        st.warning(f"Premissa prudencial não atendida: {rs['escopo_total'] - rs['escopo_validos']} de "
                   f"{rs['escopo_total']} modelos do escopo estão sem validação ou com validação vencida, e recebem o "
                   f"FPR de pior caso ({C.pct(rs['parcela_capital_descumprimento'])} do capital). Detalhes na aba "
                   "Premissa e RAS.")
    k = st.columns(5)
    caixa(k[0], "Capital de risco de modelo (R$ mi)", C.num(bruto["addon"]))
    caixa(k[1], "RWA de risco de modelo (R$ mi)", C.num(bruto["rwa"]))
    caixa(k[2], "FPR médio", C.pct(bruto["fpr_medio"]))
    caixa(k[3], "Cobertura de validação", C.pct(out["cobertura"]))
    caixa(k[4], "Capital vindo de modelos sem validação", C.pct(bruto["parcela_nao_validados"]))
    st.write("")
    st.subheader("Resumo por risco")
    st.dataframe(resumo.style.format({"Exposição (R$ mi)": C.num, "RWA (R$ mi)": C.num, "Capital (R$ mi)": C.num,
                                      "Cobertura de validação": C.pct, "FPR médio": C.pct,
                                      "Parcela do capital sem validação": C.pct}),
                 hide_index=True, width="stretch")

    st.subheader("Cálculo por modelo")

    def pinta(row):
        bg, fg = C.COR_FAIXA[row["Faixa"]]
        return [f"background-color:{bg};color:{fg};font-weight:bold" if c in ("Cor", "Nota efetiva") else
                ("color:#C00000;font-weight:bold" if c == "Situação" and row["Situação"] != "Vigente" else "")
                for c in row.index]

    dfmt = lambda d: d.strftime("%d/%m/%Y") if d else "-"
    st.dataframe(datas(det).style.apply(pinta, axis=1).format(
        {"FPR da faixa": C.pct, "λ tier": C.num, "λ nível": C.num,
         "FPR de risco de modelo": C.pct, "Exposição (R$ mi)": C.num, "RWA (R$ mi)": C.num,
         "Capital (R$ mi)": C.num}, na_rep="-"), hide_index=True, width="stretch", height=min(38 * (len(det) + 1), 720))
    st.caption("FPR de risco de modelo = FPR da faixa × λ tier × λ nível. RWA = FPR × exposição. Capital = 8% × RWA.")

    if len(det):
        d = det.sort_values("Capital (R$ mi)")
        fig = go.Figure(go.Bar(x=d["Capital (R$ mi)"], y=d["ID"] + "  " + d["Carteira ou família"], orientation="h",
                               marker_color=[C.COR_FAIXA[f][0] for f in d["Faixa"]],
                               marker_line_color="#333", marker_line_width=0.5,
                               text=[C.num(v) for v in d["Capital (R$ mi)"]], textposition="outside",
                               hovertext=d["Cor"]))
        fig.update_layout(title="Capital por modelo (cor = faixa da régua)", height=30 * len(d) + 120,
                          margin=dict(l=10, r=60, t=50, b=30), xaxis_title="R$ milhões", plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")

    st.download_button("Baixar cálculo em Excel", C.excel(det, resumo, C.regua_df(p), edit),
                       "capital_risco_modelo_calculo.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------------------------------------------------------- premissa e RAS
with aba_ras:
    st.markdown("O **capital** cobre a fatia material do inventário (escopo desta ferramenta). A **RAS** mede a "
                "cobertura de validação do inventário total, incluindo os modelos que não entram no capital. Como "
                "premissa prudencial, todos os modelos do escopo do capital devem estar com validação ativa e em dia; "
                "se não estiverem, recebem o FPR de pior caso da régua.")
    st.subheader("Premissa prudencial do escopo do capital")
    k = st.columns(3)
    caixa(k[0], "Modelos no escopo com validação em dia", f"{rs['escopo_validos']} de {rs['escopo_total']}")
    caixa(k[1], "Situação da premissa", "Atendida" if rs["premissa_ok"] else "Não atendida")
    caixa(k[2], "Capital de modelos fora da premissa (R$ mi)",
          f"{C.num(rs['capital_descumprimento'])} ({C.pct(rs['parcela_capital_descumprimento'])})")
    st.write("")
    if rs["premissa_ok"]:
        st.success("Todos os modelos do escopo do capital estão com validação ativa e em dia.")
    else:
        d = rs["descumprimento"][["ID", "Risco", "Carteira ou família", "Tier", "Nota informada", "Última validação",
                                  "Válida até", "Situação", "Exposição (R$ mi)", "Capital (R$ mi)"]]
        dfmt2 = lambda x: x.strftime("%d/%m/%Y") if x else "-"
        st.dataframe(datas(d).style.format({"Exposição (R$ mi)": C.num,
                                     "Capital (R$ mi)": C.num}, na_rep="-"), hide_index=True, width="stretch")
        st.caption("Esses modelos recebem o FPR de pior caso enquanto a validação não for concluída ou renovada.")

    st.subheader("RAS: cobertura de validação do inventário total")
    k = st.columns(4)
    caixa(k[0], "Modelos no inventário total", f"{rs['inv_total']}")
    caixa(k[1], "Validados e em dia", f"{rs['inv_validos']}")
    caixa(k[2], "Cobertura do inventário", C.pct(rs["cobertura"]))
    caixa(k[3], "Mínimo da RAS", C.pct(rs["cobertura_min"]))
    st.write("")
    tab = pd.DataFrame([
        {"Grupo": "Escopo do capital", "Modelos": rs["escopo_total"], "Validados e em dia": rs["escopo_validos"],
         "Cobertura": rs["escopo_validos"] / rs["escopo_total"] if rs["escopo_total"] else 1.0},
        {"Grupo": "Fora do escopo do capital", "Modelos": rs["fora_total"], "Validados e em dia": rs["fora_validos"],
         "Cobertura": rs["fora_cobertura"]},
        {"Grupo": "Inventário total", "Modelos": rs["inv_total"], "Validados e em dia": rs["inv_validos"],
         "Cobertura": rs["cobertura"]}])
    st.dataframe(tab.style.format({"Cobertura": C.pct}), hide_index=True, width="content")
    if rs["ras_ok"]:
        st.success(f"Dentro do apetite: cobertura de {C.pct(rs['cobertura'])}, acima do mínimo de "
                   f"{C.pct(rs['cobertura_min'])}. Folga de {rs['folga_modelos']} modelo(s) validado(s).")
    else:
        st.error(f"Fora do apetite: cobertura de {C.pct(rs['cobertura'])}, abaixo do mínimo de "
                 f"{C.pct(rs['cobertura_min'])}. Faltam {-rs['folga_modelos']} modelo(s) validado(s).")
    st.caption("Os modelos fora do escopo não recebem capital próprio, por baixa materialidade (premissa assumida); "
               "são acompanhados pela RAS e pelo plano de validação. A cobertura do escopo vem do inventário editável; "
               "a dos demais modelos, dos parâmetros da barra lateral.")

# ----------------------------------------------------------------------------- régua e matrizes
def matriz_html(titulo: str, fpr: dict) -> str:
    linhas = ""
    for lab, linha in zip(["Robusto", "Satisfatório", "Insatisfatório"], C.MATRIZ):
        cel = "".join(f"<td class='c' style='background:{C.COR_NOTA[n][0]};color:{C.COR_NOTA[n][1]}'>"
                      f"<b>{n}</b><span>{C.pct(fpr[n])}</span></td>" for n in linha)
        linhas += f"<tr><td class='lab'>{lab}</td>{cel}</tr>"
    return (f"<div class='mt'>{titulo}</div><table class='mx'><tr><th></th><th>Insatisf.</th><th>Satisf.</th>"
            f"<th>Robusto</th></tr>{linhas}</table>"
            f"<div class='na'>Sem validação ou vencida: <b>{C.pct(fpr['n/a'])}</b></div>"
            "<div style='text-align:center;font-size:11px;color:#555;margin-top:2px'>"
            "linhas: quantitativo · colunas: qualitativo</div>")


with aba_mat:
    st.subheader("Régua padronizada (tier G; mercado nível 3)")
    rg = C.regua_df(p)

    def pinta_faixa(row):
        bg, fg = C.COR_FAIXA[row["Faixa"]]
        return [f"background-color:{bg};color:{fg};font-weight:bold" if c == "Cor (notas)" else "" for c in row.index]

    st.dataframe(rg.style.apply(pinta_faixa, axis=1).format({"FPR crédito": C.pct, "FPR mercado": C.pct}),
                 hide_index=True, width="content")
    st.caption("Cada faixa recebe o maior FPR calibrado das suas notas, arredondado para cima ao múltiplo de 0,5 p.p. "
               "O racional completo está na aba Racional dos FPRs.")

    st.subheader("Matrizes de FPR por situação")
    st.caption(f"Ajustes aplicados: λ tier M = {C.num(p.lam_tier['M'])}; λ nível 2 = {C.num(p.lam_nivel['N2'])}.")
    cb = list(C.combos(p).items())
    for i in range(0, len(cb), 3):
        cols = st.columns(3)
        for col, (nome, (risco, lam)) in zip(cols, cb[i:i + 3]):
            col.markdown(matriz_html(f"{nome} (λ = {C.num(lam)})", C.matriz_fpr(p, risco, lam)),
                         unsafe_allow_html=True)
        st.write("")

# ----------------------------------------------------------------------------- cenários
with aba_cen:
    st.markdown("Capital do inventário atual se todos os modelos, ou só os pendentes (sem validação ou vencidos), "
                "fossem validados na data-base com cada nota.")
    cen, atual, teto = C.cenarios(p, inv, data_base)
    k = st.columns(3)
    caixa(k[0], "Inventário atual (R$ mi)", C.num(atual))
    caixa(k[1], "Todos validados com nota A (R$ mi)", C.num(cen.iloc[0]["Todos validados com a nota"]))
    caixa(k[2], "Nenhum modelo validado (R$ mi)", C.num(teto))
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(cen.style.apply(lambda r: [f"background-color:{C.COR_NOTA[r['Nota']][0]};"
                                                f"color:{C.COR_NOTA[r['Nota']][1]};font-weight:bold"
                                                if c == "Nota" else "" for c in r.index], axis=1)
                     .format({"Todos validados com a nota": C.num, "Só os pendentes validados com a nota": C.num}),
                     hide_index=True, width="stretch", height=370)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=cen["Nota"], y=cen["Todos validados com a nota"], name="Todos validados com a nota",
                    marker_color="#101448")
        fig.add_bar(x=cen["Nota"], y=cen["Só os pendentes validados com a nota"],
                    name="Só os pendentes validados", marker_color="#B08D3C")
        fig.add_hline(y=atual, line_dash="dash", line_color="#C00000",
                      annotation_text=f"Atual: {C.num(atual)}", annotation_position="top left")
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white",
                          yaxis_title="Capital (R$ mi)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- racional
with aba_rac:
    st.markdown("Relatório com o racional de cálculo dos FPRs. Os números são recalculados pelo motor da metodologia.")
    st.download_button("Baixar relatório (HTML)", C.relatorio_html(p, resumo, data_base, rs),
                       "racional_fpr_risco_modelo.html", "text/html")
    p0 = p
    for s in C.racional(p):
        st.subheader(s["titulo"])
        st.markdown(s["texto"])
        if s.get("formula"):
            st.latex(s["formula"])
        if s.get("tabela") is not None:
            st.dataframe(s["tabela"], hide_index=True, width="content")
        if s.get("nota"):
            st.caption(s["nota"])
        if s["titulo"].startswith("5."):
            cur = C.curva_pert(p0)
            fb = R.fator_base(p0, "RC")
            fig = go.Figure()
            for r, cor, dash in (("RC", "#101448", "solid"), ("RM", "#888", "dash")):
                x, y, _ = cur[r]
                fig.add_scatter(x=100 * x, y=y / 100, name=C.NOME_RISCO[r], line=dict(color=cor, dash=dash))
            x0, y0, c0 = cur["RC"]
            from scipy.stats import beta as B
            for n in R.NOTAS:
                xv = fb[n]["fator"]
                yv = B.pdf((xv - c0["min"]) / (c0["max"] - c0["min"]), c0["alpha"], c0["beta"]) / (c0["max"] - c0["min"])
                fig.add_scatter(x=[100 * xv], y=[yv / 100], mode="markers+text", text=[n], textposition="top center",
                                marker=dict(size=11, color=C.COR_NOTA[n][0], line=dict(color="#333", width=1)),
                                showlegend=False)
            fig.update_layout(title="Cauda Beta-PERT e posição das notas (crédito, tier G)", height=380,
                              xaxis_title="Fator de capital (% da exposição)", yaxis_title="Densidade",
                              plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=30))
            st.plotly_chart(fig, width="stretch")
        if s["titulo"].startswith("6."):
            pc = R.Parametros(abordagem="calibrada")
            cal = R.fpr_calibrado(pc, "RC")
            ordem = sorted(cal, key=cal.get)
            rgr = R.regua(p0)["RC"]
            fig = go.Figure()
            fig.add_bar(x=ordem, y=[100 * cal[n] for n in ordem], name="FPR calibrado", marker_color="#101448")
            fig.add_bar(x=ordem, y=[100 * rgr[R.FAIXA[n]] for n in ordem], name="Régua padronizada",
                        marker_color=[C.COR_NOTA[n][0] for n in ordem], marker_line_color="#333",
                        marker_line_width=0.8)
            fig.update_layout(barmode="group", title="Crédito, tier G: FPR calibrado e régua por nota (%)",
                              height=360, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=30),
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- memorial de cálculo
with aba_mem:
    st.markdown(
        "Memorial de cálculo em formato de paper (PDF), gerado a partir da base de entrada atual, seja o inventário "
        "de exemplo ou o editado na aba Inventário, e dos parâmetros da barra lateral. O documento traz:")
    st.markdown(
        "- **Escopo e base de entrada:** modelos, tiers, níveis, notas, datas de validação e exposições.\n"
        "- **Racional da modelagem:** probabilidade de falha (Basileia, 1996), magnitude pela fórmula IRB, dois "
        "canais de erro e peso ω, cauda Beta-PERT, derivação da régua padronizada e fatores λ.\n"
        "- **Execução:** situação de validação, FPR, RWA e capital por modelo, exemplos passo a passo, "
        "consolidação e cenários.\n"
        "- **Premissa prudencial e RAS, governança e limitações, referências teóricas e regulatórias.**")
    assinatura = (edit.to_csv(index=False), str(data_base), lam_m, lam_n2, int(tot_fora), int(val_fora), cob_min,
                  usar_gini, tuple(ref_gini.values()), sig_man)
    if st.button("Gerar memorial de cálculo", type="primary"):
        with st.spinner("Gerando o memorial..."):
            cen_m, atual_m, teto_m = C.cenarios(p, inv, data_base)
            st.session_state["memorial"] = (assinatura, MEM.gerar_pdf(p, inv, data_base, out, rs, cen_m, atual_m,
                                                                      teto_m, info_gini))
    mem = st.session_state.get("memorial")
    if mem and mem[0] == assinatura:
        st.success(f"Memorial gerado para a data-base {data_base.strftime('%d/%m/%Y')}: {len(inv)} modelos, "
                   f"capital de R$ {C.num(out['bruto']['addon'])} milhões.")
        st.download_button("Baixar memorial (PDF)", data=mem[1],
                           file_name=f"memorial_capital_risco_modelo_{data_base.strftime('%Y%m%d')}.pdf",
                           mime="application/pdf")
    elif mem:
        st.info("A base de entrada ou os parâmetros mudaram desde a última geração. Gere o memorial novamente.")

# ----------------------------------------------------------------------------- Gini e P de crédito
with aba_gini:
    st.markdown(
        "Em crédito, a probabilidade de falha material de cada nível quantitativo é a chance de o Gini do modelo "
        "ficar abaixo do mínimo em um ano, dada a volatilidade do Gini observada no monitoramento:")
    st.latex(r"P(\text{nível}) = \Phi\!\left(-\,\frac{\text{Gini}_{\text{nível}} - \text{Gini}_{\text{mín}}}"
             r"{\sigma}\right)")
    st.markdown("A mesma escala vale para os dois eixos da nota de crédito. Mercado mantém a escala de Basileia "
                "(1996). O peso ω de crédito é recalculado no centro do intervalo que reproduz a matriz de cores.")
    k = st.columns(4)
    caixa(k[0], "Fonte do P de crédito", "Gini" if info_gini["usar"] else "Basileia 1996")
    caixa(k[1], "σ usada (p.p./ano)", C.num(100 * info_gini["sigma"]) if info_gini["sigma"] else "-")
    caixa(k[2], "Modelos com σ informada", str(info_gini["n_sigma"]))
    caixa(k[3], "ω de crédito", C.num(R.k_risco(p, "RC")))
    st.write("")
    st.markdown("**Probabilidade de falha por nível**")
    pg = R.p_gini(info_gini["sigma"], ref_gini) if info_gini["sigma"] else None
    tab_p = pd.DataFrame([{"Nível": l, "Gini de referência": C.pct(ref_gini[l]),
                           "z": C.num((ref_gini[l] - ref_gini["minimo"]) / info_gini["sigma"])
                           if info_gini["sigma"] else "-",
                           "P pelo Gini": C.pct(pg[l]) if pg else "-", "P Basileia 1996": C.pct(p.P[l])}
                          for l in R.NIVEIS])
    st.dataframe(tab_p, hide_index=True, width="stretch")
    if info_gini["usar"]:
        lo, hi = info_gini["intervalo"]
        st.caption(f"Intervalo de ω que reproduz a matriz de cores em crédito: {C.num(lo)} a {C.num(hi)}. "
                   f"Em mercado: ω = {C.num(p.k_quali)}.")
    st.markdown("**Régua de crédito: Gini e Basileia 1996**")
    rg_g, rg_b = R.regua(p)["RC"], R.regua(R.com_gini(p, None))["RC"]
    st.dataframe(pd.DataFrame([{"Faixa": C.NOME_FAIXA[f], "FPR vigente": C.pct(rg_g[f]),
                                "FPR com Basileia 1996": C.pct(rg_b[f])} for f in R.FAIXAS]),
                 hide_index=True, width="stretch")

    st.markdown("**Distância à reprovação por modelo**")
    st.caption("z = (Gini atual − Gini mínimo) / σ do modelo; P individual = Φ(−z). Nível pelo Gini com cortes nos "
               "pontos médios entre as referências. Informativo: o capital usa o P por nível.")
    dg = C.diagnostico_gini(inv, ref_gini, data_base)
    if len(dg):
        vis = dg.copy()
        vis["Gini atual"] = [C.pct(x) if pd.notna(x) else "-" for x in dg["Gini atual"]]
        vis["σ do Gini"] = [C.num(100 * x) if pd.notna(x) else "-" for x in dg["σ do Gini"]]
        vis["Distância à reprovação (z)"] = [C.num(x) if pd.notna(x) else "-" for x in dg["Distância à reprovação (z)"]]
        vis["P individual"] = [C.pct(x) if pd.notna(x) else "-" for x in dg["P individual"]]
        vis = vis.rename(columns={"σ do Gini": "σ do Gini (p.p./ano)"})
        st.dataframe(vis, hide_index=True, width="stretch")
        if (dg["Coerente"] == "Não").any():
            st.info("Há modelos cujo nível pelo Gini difere do nível quantitativo da nota: "
                    + ", ".join(dg.loc[dg["Coerente"] == "Não", "ID"]) + ".")

    st.markdown("**Série de Gini do monitoramento**")
    st.caption("CSV ou Excel com colunas ID, Data e Gini (%). Para cada modelo, o app calcula o Gini atual (última "
               "observação) e a σ anual (desvio-padrão das variações entre observações consecutivas × √ observações "
               "por ano), e pode preencher essas colunas no inventário.")
    with open(C.SERIE_EXEMPLO, "rb") as fh:
        st.download_button("Baixar série de exemplo (CSV)", fh.read(), file_name="serie_gini_exemplo.csv",
                           mime="text/csv")
    arq = st.file_uploader("Carregar série de Gini", type=["csv", "xlsx", "xls"])
    serie = None
    try:
        serie = C.ler_serie(arq) if arq is not None else C.ler_serie(C.SERIE_EXEMPLO)
    except Exception as e:  # noqa: BLE001
        st.error(f"Não foi possível ler a série: {e}")
    if serie is not None and len(serie):
        st.caption("Série carregada." if arq is not None else "Exibindo a série de exemplo.")
        est = C.gini_da_serie(serie)
        vis = est.copy()
        vis["Início"] = [d.strftime("%d/%m/%Y") for d in est["Início"]]
        vis["Fim"] = [d.strftime("%d/%m/%Y") for d in est["Fim"]]
        vis[C.COL_GINI] = [C.num(x) for x in est[C.COL_GINI]]
        vis[C.COL_SIGMA] = [C.num(x) if pd.notna(x) else "-" for x in est[C.COL_SIGMA]]
        st.dataframe(vis, hide_index=True, width="stretch")
        if (est["Observações"] < 3).any():
            st.warning("Modelos com menos de 3 observações ficam sem σ.")
        fig = go.Figure()
        for id_, g in serie.groupby("ID", sort=False):
            fig.add_trace(go.Scatter(x=g["Data"], y=g["Gini (%)"], mode="lines+markers", name=id_,
                                     marker=dict(size=4)))
        fig.add_hline(y=100 * ref_gini["minimo"], line_dash="dash", line_color="#C00000",
                      annotation_text=f"Mínimo: {C.num(100 * ref_gini['minimo'])}%", annotation_position="bottom left")
        fig.update_layout(height=380, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                          yaxis_title="Gini (%)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        if st.button("Aplicar Gini atual e σ ao inventário", type="primary"):
            novo, msgs = C.aplicar_serie(edit, est)
            st.session_state["inv"] = novo
            st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
            st.session_state["msg_gini"] = msgs
            st.rerun()
    for m_ in st.session_state.pop("msg_gini", []):
        st.success(m_)
