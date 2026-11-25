"""App Streamlit: capital interno de risco de modelo pelo desvio medido em indicadores internos de performance."""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calculo as C
import interno as N
import memorial as MEM
import rmcap as R
import rmint as I

st.set_page_config(page_title="Capital de Risco de Modelo", layout="wide")

st.markdown("""<style>
.metric-box{border:1px solid #d0d4e4;border-radius:6px;padding:10px 14px;background:#f7f8fc;height:112px;
display:flex;flex-direction:column;justify-content:space-between;box-sizing:border-box}
.metric-box .l{font-size:13px;color:#555;line-height:1.3;min-height:2.6em;overflow:hidden}
.metric-box .v{font-size:24px;font-weight:700;color:#101448;line-height:1.1;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
.mx{border-collapse:separate;border-spacing:5px;margin:auto}
.mx td.c{width:96px;height:62px;text-align:center;border-radius:6px;border:1px solid #444;font-family:Arial}
.mx td.c b{font-size:20px;display:block}.mx td.c span{font-size:13px}
.mx td.lab{font-size:11px;color:#444;text-align:right;padding-right:4px}
.mx th{font-size:11px;color:#444;font-weight:normal}
.na{margin:4px auto 0;width:306px;text-align:center;border-radius:6px;background:#7F7F7F;color:#fff;padding:5px;
font-family:Arial;font-size:13px}
.mt{text-align:center;font-weight:700;color:#101448;margin-bottom:2px}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### Parâmetros")
    data_base = st.date_input("Data-base", value=R.DATA_BASE, format="DD/MM/YYYY")
    alfa = st.number_input("Confiança do desvio em um ano (%)", 90.0, 99.99, 99.9, 0.1, format="%.2f",
                           help="Quantil do desvio do modelo. Padrão: 99,9%, o nível do IRB.") / 100
    conf_s = st.number_input("Confiança implícita da sensibilidade N3 (%)", 60.0, 99.0, 90.0, 1.0, format="%.2f",
                             help="Nível de confiança atribuído às premissas razoavelmente possíveis da divulgação do "
                                  "CPC 46 (padrão: 90%, como no ajuste de avaliação prudente).") / 100
    st.caption("Validade da validação: 12 meses (tier G) e 36 meses (tier M). O horizonte do desvio de crédito é o "
               "ciclo do tier (1 ou 3 anos); em mercado, 1 ano. Validação vencida ou ausente recebe nota de risco κ de 100%.")
    st.markdown("### Apetite a risco (RAS)")
    cob_min = st.number_input("Cobertura mínima de validação do inventário (%)", 0.0, 100.0, 80.0, 1.0,
                              format="%.2f") / 100
    tot_fora = st.number_input("Modelos fora do escopo do capital", 0, 10000, 60, 1)
    val_fora = st.number_input("Desses, validados e em dia", 0, 10000, 50, 1)
    if val_fora > tot_fora:
        st.warning("Validados fora do escopo maior que o total; ajustado ao total.")
        val_fora = tot_fora
    if st.button("Restaurar inventário de exemplo"):
        st.session_state.pop("inv", None)
        st.session_state.pop("serie_cred", None)
        st.session_state.pop("serie_ipv", None)
        st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
        st.rerun()

p = I.ParamInt(alfa=alfa, conf_sens=conf_s)
if "inv" not in st.session_state:
    st.session_state["inv"] = N.inventario_exemplo()
if "serie_ipv" not in st.session_state:
    st.session_state["serie_ipv"] = N.serie_exemplo_ipv()
    st.session_state["serie_ipv_nome"] = "série de exemplo"
if "serie_cred" not in st.session_state:
    st.session_state["serie_cred"] = N.serie_exemplo_credito()
    st.session_state["serie_cred_nome"] = "série de exemplo"

st.title("Capital de risco de modelo")
st.caption("Pilar 2 · abordagem interna: capital = exposição × desvio do modelo no quantil α × (1 + nota de risco κ). "
           "Exposições em R$ bilhões e capital em R$ milhões. Inventário, séries e saldos ilustrativos.")

aba_inv, aba_res, aba_des, aba_mat, aba_ras, aba_cen, aba_mem = st.tabs(
    ["Inventário", "Resultado", "Desvio por modelo", "Matriz e nota de risco κ", "Premissa e RAS", "Cenários",
     "Memorial de cálculo"])

# ----------------------------------------------------------------------------- inventário
with aba_inv:
    st.markdown("Edite notas, tiers, níveis, datas de validação, exposições e indicadores de desvio. Crédito usa "
                "classe IRB e LGD, e a série de calibragem por GH (aba Desvio por modelo); mercado N2 usa a série do IPV por família de instrumento (aba Desvio por modelo); mercado N3 usa a "
                "sensibilidade do CPC 46 e, se houver, a diferença para o modelo alternativo. Os indicadores podem "
                "ser preenchidos pelas séries na aba Desvio por modelo.")
    edit = st.data_editor(
        st.session_state["inv"], key=f"editor_{st.session_state.get('editor_ver', 0)}", num_rows="dynamic",
        width="stretch", hide_index=True, height=38 * (len(st.session_state["inv"]) + 3),
        column_config={
            "Risco": st.column_config.SelectboxColumn(options=["Crédito", "Mercado"], required=True),
            "Tier": st.column_config.SelectboxColumn(options=["G", "M"], required=True),
            "Nível (mercado)": st.column_config.SelectboxColumn(options=["N2", "N3"]),
            "Nota": st.column_config.SelectboxColumn(options=C.NOTAS9 + [C.SEM_VAL], required=True),
            "Última validação": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Exposição (R$ bi)": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
            "Modelo": st.column_config.TextColumn(width="medium"),
            N.COL_CLASSE: st.column_config.SelectboxColumn(options=list(I.CLASSES), help="Só crédito."),
            N.COL_LGD: st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=100.0,
                                                     help="Só crédito."),
            N.COL_SENS: st.column_config.NumberColumn(format="%.2f", min_value=0.0,
                                                      help="Só N3: efeito no valor justo de premissas alternativas "
                                                           "razoavelmente possíveis (CPC 46, item 93(h)(ii))."),
            N.COL_ALT: st.column_config.NumberColumn(format="%.2f", min_value=0.0,
                                                     help="Só N3: diferença para o modelo alternativo, % do VJ."),
        })

inv, avisos = N.para_modelos(edit, st.session_state["serie_cred"], st.session_state["serie_ipv"])
out = N.calcular(p, inv, data_base)
det, resumo = out["detalhe"], out["resumo"]
rs = C.ras(det, int(tot_fora), int(val_fora), cob_min)
ref = N.referencia_regua(inv, data_base)
with aba_inv:
    for a in avisos:
        st.warning(a)
    falta = det[det["Fonte do desvio"] != "Medido"] if len(det) else det
    for _, r in falta.iterrows():
        st.info(f"{r['ID']}: sem indicador de desvio; usado {r['Fonte do desvio'].lower()}.")


def datas(df: pd.DataFrame) -> pd.DataFrame:
    out_ = df.copy()
    for c in ("Última validação", "Válida até"):
        if c in out_:
            out_[c] = [x.strftime("%d/%m/%Y") if x is not None and pd.notna(x) else "-" for x in out_[c]]
    return out_


def caixa(col, rotulo, valor):
    col.markdown(f"<div class='metric-box'><div class='l'>{rotulo}</div><div class='v'>{valor}</div></div>",
                 unsafe_allow_html=True)


def fmt(df: pd.DataFrame, pcts=(), nums=(), dec=()):
    f = {c: C.pct for c in pcts if c in df}
    f.update({c: C.num for c in nums if c in df})
    f.update({c: C.num for c in dec if c in df})
    return df.style.format(f, na_rep="-")


# ----------------------------------------------------------------------------- resultado
with aba_res:
    if not rs["premissa_ok"]:
        st.warning(f"Premissa prudencial não atendida: {rs['escopo_total'] - rs['escopo_validos']} de "
                   f"{rs['escopo_total']} modelos do escopo estão sem validação ou com validação vencida e recebem "
                   f"nota de risco κ de 100% ({C.pct(rs['parcela_capital_descumprimento'])} do capital).")
    k = st.columns(4)
    caixa(k[0], "Capital interno de risco de modelo (R$ mi)", C.num(out["capital"]))
    caixa(k[1], "Taxa de capital sobre a exposição", C.pct(out["capital"] / out["exposicao"] if out["exposicao"] else 0))
    caixa(k[2], "Capital de modelos sem validação vigente", C.pct(rs["parcela_capital_descumprimento"]))
    caixa(k[3], "Referência: régua padronizada (R$ mi)", C.num(ref))
    st.write("")
    st.subheader("Por família")
    st.dataframe(fmt(resumo, pcts=["Taxa de capital"], nums=["Exposição (R$ bi)", "Capital (R$ mi)",
                                                           "Capital sem validação vigente"]),
                 hide_index=True, width="stretch")
    st.subheader("Por modelo")
    cols = ["ID", "Família", "Carteira ou família", "Tier", "Nota efetiva", "Horizonte (anos)", "Fonte do desvio",
            "Desvio D", "Nota de risco κ", "Taxa de capital", "Exposição (R$ bi)", "Capital (R$ mi)"]
    st.dataframe(fmt(det[cols], pcts=["Desvio D", "Nota de risco κ", "Taxa de capital"],
                     nums=["Exposição (R$ bi)", "Capital (R$ mi)", "Horizonte (anos)"]),
                 hide_index=True, width="stretch")
    if len(det):
        d2 = det.sort_values("Capital (R$ mi)", ascending=True)
        cor = {I.FAMILIAS["RC"]: "#101448", I.FAMILIAS["N2"]: "#B08D3C", I.FAMILIAS["N3"]: "#C00000"}
        fig = go.Figure()
        for fam, c_ in cor.items():
            s_ = d2[d2["Família"] == fam]
            fig.add_bar(y=s_["ID"], x=s_["Capital (R$ mi)"], orientation="h", name=fam, marker_color=c_)
        fig.update_layout(height=520, barmode="stack", plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                          xaxis_title="Capital (R$ mi)", legend=dict(orientation="h", y=-0.12),
                          yaxis=dict(categoryorder="array", categoryarray=list(d2["ID"])))
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- desvio por modelo
with aba_des:
    st.markdown(f"O desvio D é o erro do modelo no quantil de {C.pct(p.alfa)} em um ano, em % da exposição, no "
                "horizonte h: em crédito, o ciclo de validação do tier (1 ano no G, 3 anos no M); em mercado, 1 ano, "
                "porque o IPV é controle mensal e a sensibilidade do N3 já é uma faixa de premissas plausíveis.")
    tab_c, tab_2, tab_3 = st.tabs(["Crédito: calibragem", "Mercado N2: IPV", "Mercado N3: sensibilidade"])
    with tab_c:
        st.markdown("Para cada GH do modelo, compara-se a PD do GH com a BadRate observada no mesmo GH. Só conta o "
                    "desvio que não é ruído amostral: o viés entra se a BadRate acumulada sair do intervalo de "
                    "Jeffreys de 95%, e a volatilidade entra líquida da variância binomial, se o teste de "
                    "sobredispersão (χ² a 95%) indicar variação além do ruído.")
        st.latex(r"d_{g,t}=BR_{g,t}-PD_g,\qquad \sigma_g^2=\max\!\left(0,\ s^2_{d_g}-\overline{\tfrac{PD_g(1-PD_g)}"
                 r"{n_{g,t}}}\right),\qquad PD_g^{*}=PD_g+\max\!\left(0,\ b_g+z_\alpha\,\sigma_g\sqrt{h}\right)")
        st.latex(r"D_{RC}=\sum_g w_g\,\max\left\{K(PD_g^{*})-K(PD_g);\ (PD_g^{*}-PD_g)\cdot LGD\right\}")
        st.caption("b_g: viés do GH (BadRate acumulada − PD), zero se dentro do intervalo de Jeffreys. w_g: "
                   "participação do GH no saldo. K: fórmula IRB da classe do modelo.")
        sub = det[det["Risco"] == "Crédito"]
        if len(sub):
            ind = pd.DataFrame([{"ID": m.id, N.COL_CLASSE: m.ind.get("classe"), "LGD": m.ind.get("lgd"),
                                 "GHs": len(m.ind.get("ghs") or [])} for m in inv if m.risco == "RC"])
            v = ind.merge(sub[["ID", "Horizonte (anos)", "PD média (crédito)", "PD estressada média",
                               "Desvio capital IRB", "Desvio provisão", "Critério", "Desvio D"]], on="ID")
            st.dataframe(fmt(v, pcts=["LGD", "PD média (crédito)", "PD estressada média", "Desvio capital IRB",
                                      "Desvio provisão", "Desvio D"], nums=["Horizonte (anos)"]),
                         hide_index=True, width="stretch")
        st.markdown(f"**Série de calibragem por GH** ({st.session_state.get('serie_cred_nome', '')}): ID, Data, GH, "
                    "PD do GH (%), Contratos, Bads, Saldo (R$ mi)")
        c1_, c2_ = st.columns([1, 3])
        with open(N.SERIE_CRED, "rb") as fh:
            c1_.download_button("Baixar série de exemplo", fh.read(), "serie_credito_exemplo.csv", "text/csv")
        arq = c2_.file_uploader("Carregar série de calibragem", type=["csv", "xlsx", "xls"], key="up_cred")
        if arq is not None and st.session_state.get("serie_cred_nome") != arq.name:
            try:
                st.session_state["serie_cred"] = N.ler_serie_credito(arq)
                st.session_state["serie_cred_nome"] = arq.name
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Não foi possível ler a série: {e}")
        serie = st.session_state["serie_cred"]
        est_gh = N.estat_credito(serie)
        sem = [m.id for m in inv if m.risco == "RC" and m.id not in est_gh]
        if sem:
            st.info("Modelos de crédito sem GH na série: " + ", ".join(sem) + ".")
        tg = N.tabela_gh(est_gh)
        if len(tg):
            ids = list(est_gh)
            sel = st.selectbox("Modelo", ids, key="sel_gh")
            vg = tg[tg["ID"] == sel].drop(columns=["ID"])
            st.dataframe(fmt(vg, pcts=["PD do GH", "BadRate acumulada", "IC Jeffreys inf.", "IC Jeffreys sup.",
                                       "Viés considerado", "σ observada", "σ binomial", "σ líquida"],
                             nums=["Contratos médios", "Saldo (R$ mi)"]), hide_index=True, width="stretch")
            gs = serie[serie["ID"] == sel]
            fig = go.Figure()
            paleta = ["#101448", "#B08D3C", "#C00000", "#2E7D32", "#6A1B9A", "#00838F"]
            for k_, (gh, g) in enumerate(gs.groupby("GH")):
                cor_ = paleta[k_ % len(paleta)]
                fig.add_trace(go.Scatter(x=g["Data"], y=100 * g["Bads"] / g["Contratos"], mode="lines+markers",
                                         name=f"GH {gh}: BadRate", line=dict(color=cor_), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=g["Data"], y=g["PD do GH (%)"], mode="lines", name=f"GH {gh}: PD",
                                         line=dict(color=cor_, dash="dash"), showlegend=False))
            fig.update_layout(height=380, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                              yaxis_title="% (linha cheia: BadRate; tracejada: PD do GH)",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
    with tab_2:
        st.markdown("Para cada família de instrumento do modelo, o desvio do IPV é a diferença entre a marcação do "
                    "modelo e a fonte independente (Anbima, B3, contraparte), em % do valor justo. Só conta o "
                    "excesso sobre a tolerância da fonte (spread, dispersão entre fontes, intervalo indicativo); o "
                    "viés entra apenas se o teste t rejeitar média zero a 95%.")
        st.latex(r"e_{f,t}=\mathrm{sinal}(d_{f,t})\,\max\left(0,\ |d_{f,t}|-\tau_{f,t}\right),\qquad "
                 r"e_f\sim N(b_f,\ \sigma_f^2),\qquad \Pr\left(|e_f|\le D_f\right)=\alpha")
        st.latex(r"D_{N2}=\sum_f w_f\,D_f")
        st.caption("b_f: média do excesso se significativa, zero caso contrário. σ_f: desvio-padrão do excesso. w_f: "
                   "participação da família no valor justo. Horizonte de 1 ano.")
        sub = det[(det["Risco"] == "Mercado") & (det["Nível"] == "N2")]
        if len(sub):
            ind = pd.DataFrame([{"ID": m.id, "Famílias": len(m.ind.get("fams") or [])} for m in inv
                                if m.risco == "RM" and m.nivel == "N2"])
            v = ind.merge(sub[["ID", "Carteira ou família", "Fonte do desvio", "Desvio D"]], on="ID")
            st.dataframe(fmt(v, pcts=["Desvio D"]), hide_index=True, width="stretch")
        st.markdown(f"**Série do IPV por família de instrumento** ({st.session_state.get('serie_ipv_nome', '')}): ID, "
                    "Data, Família de instrumento, Desvio IPV (%), Tolerância (%), Valor justo (R$ mi)")
        c1_, c2_ = st.columns([1, 3])
        with open(N.SERIE_IPV, "rb") as fh:
            c1_.download_button("Baixar série de exemplo", fh.read(), "serie_ipv_exemplo.csv", "text/csv")
        arq = c2_.file_uploader("Carregar série de IPV", type=["csv", "xlsx", "xls"], key="up_ipv")
        if arq is not None and st.session_state.get("serie_ipv_nome") != arq.name:
            try:
                st.session_state["serie_ipv"] = N.ler_serie_ipv(arq)
                st.session_state["serie_ipv_nome"] = arq.name
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Não foi possível ler a série: {e}")
        serie_m = st.session_state["serie_ipv"]
        est_m = N.estat_mercado(serie_m)
        sem = [m.id for m in inv if m.risco == "RM" and m.nivel == "N2" and m.id not in est_m]
        if sem:
            st.info("Modelos N2 sem família na série: " + ", ".join(sem) + ".")
        tf = N.tabela_familias(est_m)
        if len(tf):
            sel = st.selectbox("Modelo", list(est_m), key="sel_fam")
            vf = tf[tf["ID"] == sel].drop(columns=["ID"])
            st.dataframe(fmt(vf, pcts=["Desvio médio", "Desvio-padrão", "Tolerância", "Fora da tolerância",
                                       "Excesso médio", "Viés considerado", "σ do excesso"],
                             nums=["t", "Valor justo (R$ mi)"]), hide_index=True, width="stretch")
            gs = serie_m[serie_m["ID"] == sel]
            fig = go.Figure()
            paleta = ["#101448", "#B08D3C", "#C00000", "#2E7D32", "#6A1B9A", "#00838F"]
            for k_, (fam_, g) in enumerate(gs.groupby("Família de instrumento", sort=False)):
                cor_ = paleta[k_ % len(paleta)]
                fig.add_trace(go.Scatter(x=g["Data"], y=g["Desvio IPV (%)"], mode="lines+markers", name=fam_,
                                         line=dict(color=cor_), marker=dict(size=4)))
                for sinal in (1, -1):
                    fig.add_trace(go.Scatter(x=g["Data"], y=sinal * g["Tolerância (%)"], mode="lines",
                                             line=dict(color=cor_, dash="dot", width=1), showlegend=False,
                                             hoverinfo="skip"))
            fig.add_hline(y=0.0, line_color="#999", line_width=1)
            fig.update_layout(height=380, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                              yaxis_title="Desvio IPV, % do VJ (pontilhado: ± tolerância)",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
    with tab_3:
        st.latex(r"D=\max(s,\ a)\times\frac{z_\alpha}{z_c}")
        st.caption("s: efeito no valor justo de premissas alternativas razoavelmente possíveis para os insumos não "
                   "observáveis, divulgado pelo CPC 46 (item 93(h)(ii)); a: diferença para o modelo alternativo, "
                   "quando houver (Reg. UE 2016/101, art. 11). z_c: quantil implícito da sensibilidade "
                   f"({C.pct(p.conf_sens)}); a escala leva a sensibilidade ao quantil α.")
        sub = det[(det["Risco"] == "Mercado") & (det["Nível"] == "N3")]
        if len(sub):
            ind = pd.DataFrame([{"ID": m.id, "Sensibilidade s": m.ind.get("sens"), "Modelo alternativo a":
                                 m.ind.get("alt")} for m in inv if m.risco == "RM" and m.nivel == "N3"])
            v = ind.merge(sub[["ID", "Carteira ou família", "Horizonte (anos)", "Critério", "Desvio D"]], on="ID")
            v.insert(3, "Escala z_α/z_c", I.desvio_n3(0.01, None, 1, p)["escala"])
            st.dataframe(fmt(v, pcts=["Sensibilidade s", "Modelo alternativo a", "Desvio D"],
                             nums=["Horizonte (anos)", "Escala z_α/z_c"]), hide_index=True, width="stretch")
    for m_ in st.session_state.pop("msg", []):
        st.success(m_)

# ----------------------------------------------------------------------------- matriz e nota de risco κ
def matriz_html(titulo: str, val: dict) -> str:
    linhas = ""
    for lab, linha in zip(["Robusto", "Satisfatório", "Insatisfatório"], C.MATRIZ):
        cel = "".join(f"<td class='c' style='background:{C.COR_NOTA[n][0]};color:{C.COR_NOTA[n][1]}'>"
                      f"<b>{n}</b><span>{C.pct(val[n])}</span></td>" for n in linha)
        linhas += f"<tr><td class='lab'>{lab}</td>{cel}</tr>"
    return (f"<div class='mt'>{titulo}</div><table class='mx'><tr><th></th><th>Insatisf.</th><th>Satisf.</th>"
            f"<th>Robusto</th></tr>{linhas}</table>"
            f"<div class='na'>Sem validação ou vencida: <b>{C.pct(val['n/a'])}</b></div>"
            "<div style='text-align:center;font-size:11px;color:#555;margin-top:2px'>"
            "linhas: quantitativo · colunas: qualitativo</div>")


with aba_mat:
    st.markdown("A **nota de risco do modelo κ** é capturada pela validação independente e mede o quanto o desvio "
                "medido é confiável. Ela multiplica o desvio por (1 + κ): zero na nota A e 100% sem validação, "
                "seguindo a ordem da matriz "
                "de cores.")
    st.latex(r"e(\text{nota})=\frac{P_{q}\,\pi_{q}+\omega\,P_{ql}\,\pi_{ql}}{1+\omega},\qquad "
             r"\kappa(\text{nota})=\frac{e(\text{nota})-e(A)}{1-e(A)}")
    st.caption(f"P e π: probabilidade e proporção de magnitude por nível (Basileia, 1996, Tabelas 1 e 2); "
               f"ω = {C.num(p.omega)} (peso do eixo qualitativo). Ordem da matriz de cores: "
               f"{'atendida' if I.ordem_margem_ok(p) else 'não atendida'}.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(matriz_html("Nota de risco do modelo κ", out["margem"]), unsafe_allow_html=True)
    with c2:
        e = I.perda_relativa(p)
        tab = pd.DataFrame([{"Nota": n, "Quanti": R.GRADE[n][0] if n != "n/a" else "-",
                             "Quali": R.GRADE[n][1] if n != "n/a" else "-", "e(nota)": e[n], "κ": out["margem"][n],
                             "Multiplicador": 1 + out["margem"][n]} for n in R.NOTAS])
        st.dataframe(fmt(tab, pcts=["e(nota)", "κ"], nums=["Multiplicador"]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- premissa e RAS
with aba_ras:
    st.markdown("O **capital** cobre os modelos do escopo (behavior de PD em crédito; marcação a mercado de nível 2 "
                "e 3). A **RAS** mede a cobertura de validação do inventário total. Como premissa prudencial, todos "
                "os modelos do escopo devem estar com validação ativa e em dia; se não estiverem, recebem nota de risco κ de "
                "100% sobre o desvio medido.")
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
                                  "Válida até", "Situação", "Exposição (R$ bi)", "Capital (R$ mi)"]]
        st.dataframe(fmt(datas(d), nums=["Exposição (R$ bi)", "Capital (R$ mi)"]), hide_index=True, width="stretch")
    st.subheader("RAS: cobertura de validação do inventário total")
    k = st.columns(4)
    caixa(k[0], "Modelos no inventário total", f"{rs['inv_total']}")
    caixa(k[1], "Validados e em dia", f"{rs['inv_validos']}")
    caixa(k[2], "Cobertura do inventário", C.pct(rs["cobertura"]))
    caixa(k[3], "Mínimo da RAS", C.pct(rs["cobertura_min"]))
    st.write("")
    if rs["ras_ok"]:
        st.success(f"Dentro do apetite: cobertura de {C.pct(rs['cobertura'])}, acima do mínimo de "
                   f"{C.pct(rs['cobertura_min'])}. Folga de {rs['folga_modelos']} modelo(s) validado(s).")
    else:
        st.error(f"Fora do apetite: cobertura de {C.pct(rs['cobertura'])}, abaixo do mínimo de "
                 f"{C.pct(rs['cobertura_min'])}. Faltam {-rs['folga_modelos']} modelo(s) validado(s).")

# ----------------------------------------------------------------------------- cenários
with aba_cen:
    st.markdown("Capital se todos os modelos, ou só os pendentes (sem validação ou vencidos), fossem validados na "
                "data-base com cada nota. O desvio medido não muda; muda a nota de risco κ.")
    cen, atual, teto = N.cenarios(p, inv, data_base)
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
        fig.add_bar(x=cen["Nota"], y=cen["Só os pendentes validados com a nota"], name="Só os pendentes validados",
                    marker_color="#B08D3C")
        fig.add_hline(y=atual, line_dash="dash", line_color="#C00000", annotation_text=f"Atual: {C.num(atual)}",
                      annotation_position="top left")
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white",
                          yaxis_title="Capital (R$ mi)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- memorial
with aba_mem:
    st.markdown("Memorial de cálculo em formato de paper (PDF), gerado a partir da base de entrada atual e dos "
                "parâmetros da barra lateral: racional da modelagem por família, nota de risco κ, execução "
                "do cálculo por modelo, cenários, premissa e RAS, limitações e referências.")
    assinatura = (edit.to_csv(index=False), str(data_base), alfa, conf_s, int(tot_fora), int(val_fora), cob_min,
                  st.session_state.get("serie_cred_nome"), len(st.session_state["serie_cred"]),
                  st.session_state.get("serie_ipv_nome"), len(st.session_state["serie_ipv"]))
    if st.button("Gerar memorial de cálculo", type="primary"):
        with st.spinner("Gerando o memorial..."):
            cen_m, atual_m, teto_m = N.cenarios(p, inv, data_base)
            st.session_state["memorial"] = (assinatura, MEM.gerar_pdf_interno(p, inv, data_base, out, rs, cen_m,
                                                                              atual_m, teto_m, ref))
    mem = st.session_state.get("memorial")
    if mem and mem[0] == assinatura:
        st.success(f"Memorial gerado para a data-base {data_base.strftime('%d/%m/%Y')}: {len(inv)} modelos, capital "
                   f"de R$ {C.num(out['capital'])} milhões.")
        st.download_button("Baixar memorial (PDF)", data=mem[1],
                           file_name=f"memorial_capital_risco_modelo_{data_base.strftime('%Y%m%d')}.pdf",
                           mime="application/pdf")
    elif mem:
        st.info("A base de entrada ou os parâmetros mudaram desde a última geração. Gere o memorial novamente.")
