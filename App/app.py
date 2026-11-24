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
    alfa = st.number_input("Confiança do desvio em um ano (%)", 90.0, 99.99, 99.9, 0.1, format="%.3f",
                           help="Quantil do desvio do modelo. Padrão: 99,9%, o nível do IRB.") / 100
    conf_s = st.number_input("Confiança implícita da sensibilidade N3 (%)", 60.0, 99.0, 90.0, 1.0, format="%.3f",
                             help="Nível de confiança atribuído às premissas razoavelmente possíveis da divulgação do "
                                  "CPC 46 (padrão: 90%, como no ajuste de avaliação prudente).") / 100
    st.caption("Validade da validação: 12 meses (tier G) e 36 meses (tier M). O horizonte do desvio é o ciclo do "
               "tier (1 ou 3 anos). Validação vencida ou ausente recebe margem de 100%.")
    st.markdown("### Apetite a risco (RAS)")
    cob_min = st.number_input("Cobertura mínima de validação do inventário (%)", 0.0, 100.0, 80.0, 1.0,
                              format="%.3f") / 100
    tot_fora = st.number_input("Modelos fora do escopo do capital", 0, 10000, 60, 1)
    val_fora = st.number_input("Desses, validados e em dia", 0, 10000, 50, 1)
    if val_fora > tot_fora:
        st.warning("Validados fora do escopo maior que o total; ajustado ao total.")
        val_fora = tot_fora
    if st.button("Restaurar inventário de exemplo"):
        st.session_state.pop("inv", None)
        st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
        st.rerun()

p = I.ParamInt(alfa=alfa, conf_sens=conf_s)
if "inv" not in st.session_state:
    st.session_state["inv"] = N.inventario_exemplo()

st.title("Capital de risco de modelo")
st.caption("Pilar 2 · abordagem interna: capital = exposição × desvio do modelo no quantil α × (1 + margem de "
           "conservadorismo da nota). Valores em R$ milhões. Inventário, séries e saldos ilustrativos.")

aba_inv, aba_res, aba_des, aba_mat, aba_ras, aba_cen, aba_mem = st.tabs(
    ["Inventário", "Resultado", "Desvio por modelo", "Matriz e margem", "Premissa e RAS", "Cenários",
     "Memorial de cálculo"])

# ----------------------------------------------------------------------------- inventário
with aba_inv:
    st.markdown("Edite notas, tiers, níveis, datas de validação, exposições e indicadores de desvio. Crédito usa "
                "classe IRB, PD, LGD e a razão log(DR/PD); mercado N2 usa o desvio do IPV; mercado N3 usa a "
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
            "Exposição (R$ mi)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
            "Modelo": st.column_config.TextColumn(width="medium"),
            N.COL_CLASSE: st.column_config.SelectboxColumn(options=list(I.CLASSES), help="Só crédito."),
            N.COL_PD: st.column_config.NumberColumn(format="%.3f", min_value=0.0, max_value=100.0,
                                                    help="Só crédito: PD prevista média da carteira."),
            N.COL_LGD: st.column_config.NumberColumn(format="%.3f", min_value=0.0, max_value=100.0,
                                                     help="Só crédito."),
            N.COL_MU_C: st.column_config.NumberColumn(format="%.3f", help="Só crédito: média de ln(taxa de default "
                                                                          "observada / PD prevista)."),
            N.COL_SG_C: st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                      help="Só crédito: desvio-padrão de ln(DR/PD)."),
            N.COL_MU_I: st.column_config.NumberColumn(format="%.3f", help="Só N2: desvio médio do IPV, % do VJ."),
            N.COL_SG_I: st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                      help="Só N2: desvio-padrão do desvio do IPV."),
            N.COL_SENS: st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                      help="Só N3: efeito no valor justo de premissas alternativas "
                                                           "razoavelmente possíveis (CPC 46, item 93(h)(ii))."),
            N.COL_ALT: st.column_config.NumberColumn(format="%.3f", min_value=0.0,
                                                     help="Só N3: diferença para o modelo alternativo, % do VJ."),
        })

inv, avisos = N.para_modelos(edit)
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
                   f"margem de 100% ({C.pct(rs['parcela_capital_descumprimento'])} do capital).")
    k = st.columns(4)
    caixa(k[0], "Capital interno de risco de modelo (R$ mi)", C.num(out["capital"]))
    caixa(k[1], "Taxa de capital sobre a exposição", C.pct(out["capital"] / out["exposicao"] if out["exposicao"] else 0))
    caixa(k[2], "Capital de modelos sem validação vigente", C.pct(rs["parcela_capital_descumprimento"]))
    caixa(k[3], "Referência: régua padronizada (R$ mi)", C.num(ref))
    st.write("")
    st.subheader("Por família")
    st.dataframe(fmt(resumo, pcts=["Taxa de capital"], nums=["Exposição (R$ mi)", "Capital (R$ mi)",
                                                           "Capital sem validação vigente"]),
                 hide_index=True, width="stretch")
    st.subheader("Por modelo")
    cols = ["ID", "Família", "Carteira ou família", "Tier", "Nota efetiva", "Horizonte (anos)", "Fonte do desvio",
            "Desvio D", "Margem MC", "Taxa de capital", "Exposição (R$ mi)", "Capital (R$ mi)"]
    st.dataframe(fmt(det[cols], pcts=["Desvio D", "Margem MC", "Taxa de capital"],
                     nums=["Exposição (R$ mi)", "Capital (R$ mi)", "Horizonte (anos)"]),
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
                "horizonte do ciclo de validação do tier (h = 1 ano no tier G e 3 anos no tier M).")
    tab_c, tab_2, tab_3 = st.tabs(["Crédito: calibração", "Mercado N2: IPV", "Mercado N3: sensibilidade"])
    with tab_c:
        st.latex(r"m=\exp\!\left(\mu+z_\alpha\,\sigma\sqrt{h}\right),\qquad "
                 r"D=\max\{K(m\cdot PD)-K(PD);\ (m-1)\cdot PD\cdot LGD\}")
        st.caption("μ e σ: média e desvio-padrão de ln(taxa de default observada / PD prevista) na série do "
                   "monitoramento. K: fórmula IRB da classe (corporativo, imobiliário, rotativo, outros varejo). "
                   "O desvio é o maior entre a falta de capital e a falta de provisão se a PD real for m vezes a "
                   "prevista.")
        sub = det[det["Risco"] == "Crédito"]
        if len(sub):
            ind = pd.DataFrame([{"ID": m.id, N.COL_CLASSE: m.ind.get("classe"), "PD": m.ind.get("pd"),
                                 "LGD": m.ind.get("lgd"), "μ": m.ind.get("mu"), "σ": m.ind.get("sigma")}
                                for m in inv if m.risco == "RC"])
            v = ind.merge(sub[["ID", "Horizonte (anos)", "Fator m (crédito)", "Desvio capital IRB", "Desvio provisão",
                               "Critério", "Desvio D"]], on="ID")
            st.dataframe(fmt(v, pcts=["PD", "LGD", "Desvio capital IRB", "Desvio provisão", "Desvio D"],
                             nums=["μ", "σ", "Horizonte (anos)", "Fator m (crédito)"]), hide_index=True,
                         width="stretch")
        st.markdown("**Série do monitoramento (ID, Data, PD prevista (%), Taxa de default (%))**")
        with open(N.SERIE_CRED, "rb") as fh:
            st.download_button("Baixar série de exemplo (crédito)", fh.read(), "serie_credito_exemplo.csv", "text/csv")
        arq = st.file_uploader("Carregar série de crédito", type=["csv", "xlsx", "xls"], key="up_cred")
        try:
            serie = N.ler_serie_credito(arq) if arq is not None else N.ler_serie_credito(N.SERIE_CRED)
            est = N.estimar_credito(serie)
            st.caption("Série carregada." if arq is not None else "Exibindo a série de exemplo.")
            vis = est.copy()
            for c in ("Início", "Fim"):
                vis[c] = [d.strftime("%d/%m/%Y") for d in est[c]]
            st.dataframe(fmt(vis, nums=[N.COL_PD, N.COL_MU_C, N.COL_SG_C, "Razão DR/PD média"]), hide_index=True,
                         width="stretch")
            fig = go.Figure()
            for id_, g in serie.groupby("ID", sort=False):
                fig.add_trace(go.Scatter(x=g["Data"], y=g["Taxa de default (%)"] / g["PD prevista (%)"],
                                         mode="lines+markers", name=id_, marker=dict(size=4)))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#555")
            fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                              yaxis_title="Taxa de default / PD prevista", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            if st.button("Aplicar PD, μ e σ ao inventário", key="ap_cred", type="primary"):
                novo, msgs = N.aplicar(edit, est, [N.COL_PD, N.COL_MU_C, N.COL_SG_C], "Crédito")
                st.session_state.update(inv=novo, editor_ver=st.session_state.get("editor_ver", 0) + 1, msg=msgs)
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Não foi possível ler a série: {e}")
    with tab_2:
        st.latex(r"d\sim N(\mu,\ \sigma^2 h),\qquad D=q\ \text{tal que}\ \Pr(|d|\le q)=\alpha")
        st.caption("d: desvio do IPV, diferença entre a marcação do modelo e a fonte independente (Anbima, B3, "
                   "contraparte), em % do valor justo. O desvio é bilateral: errar para cima ou para baixo é falha.")
        sub = det[(det["Risco"] == "Mercado") & (det["Nível"] == "N2")]
        if len(sub):
            ind = pd.DataFrame([{"ID": m.id, "μ": m.ind.get("mu"), "σ": m.ind.get("sigma")} for m in inv
                                if m.risco == "RM" and m.nivel == "N2"])
            v = ind.merge(sub[["ID", "Carteira ou família", "Horizonte (anos)", "Desvio D"]], on="ID")
            st.dataframe(fmt(v, pcts=["μ", "σ", "Desvio D"], nums=["Horizonte (anos)"]), hide_index=True,
                         width="stretch")
        st.markdown("**Série do IPV (ID, Data, Desvio IPV (%))**")
        with open(N.SERIE_IPV, "rb") as fh:
            st.download_button("Baixar série de exemplo (IPV)", fh.read(), "serie_ipv_exemplo.csv", "text/csv")
        arq = st.file_uploader("Carregar série de IPV", type=["csv", "xlsx", "xls"], key="up_ipv")
        try:
            serie = N.ler_serie_ipv(arq) if arq is not None else N.ler_serie_ipv(N.SERIE_IPV)
            est = N.estimar_ipv(serie)
            st.caption("Série carregada." if arq is not None else "Exibindo a série de exemplo.")
            vis = est.copy()
            for c in ("Início", "Fim"):
                vis[c] = [d.strftime("%d/%m/%Y") for d in est[c]]
            st.dataframe(fmt(vis, nums=[N.COL_MU_I, N.COL_SG_I]), hide_index=True, width="stretch")
            fig = go.Figure()
            for id_, g in serie.groupby("ID", sort=False):
                fig.add_trace(go.Scatter(x=g["Data"], y=g["Desvio IPV (%)"], mode="lines+markers", name=id_,
                                         marker=dict(size=4)))
            fig.add_hline(y=0.0, line_dash="dash", line_color="#555")
            fig.update_layout(height=340, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=30),
                              yaxis_title="Desvio IPV (% do VJ)", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            if st.button("Aplicar μ e σ ao inventário", key="ap_ipv", type="primary"):
                novo, msgs = N.aplicar(edit, est, [N.COL_MU_I, N.COL_SG_I], "Mercado")
                st.session_state.update(inv=novo, editor_ver=st.session_state.get("editor_ver", 0) + 1, msg=msgs)
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Não foi possível ler a série: {e}")
    with tab_3:
        st.latex(r"D=\max(s,\ a)\times\frac{z_\alpha}{z_c}\times\sqrt{h}")
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

# ----------------------------------------------------------------------------- matriz e margem
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
    st.markdown("A nota da validação mede o quanto o desvio medido é confiável. A margem de conservadorismo MC "
                "multiplica o desvio por (1 + MC): zero na nota A e 100% sem validação, seguindo a ordem da matriz "
                "de cores.")
    st.latex(r"e(\text{nota})=\frac{P_{q}\,\pi_{q}+\omega\,P_{ql}\,\pi_{ql}}{1+\omega},\qquad "
             r"MC(\text{nota})=\frac{e(\text{nota})-e(A)}{1-e(A)}")
    st.caption(f"P e π: probabilidade e proporção de magnitude por nível (Basileia, 1996, Tabelas 1 e 2); "
               f"ω = {C.num(p.omega)} (peso do eixo qualitativo). Ordem da matriz de cores: "
               f"{'atendida' if I.ordem_margem_ok(p) else 'não atendida'}.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(matriz_html("Margem de conservadorismo por nota", out["margem"]), unsafe_allow_html=True)
    with c2:
        e = I.perda_relativa(p)
        tab = pd.DataFrame([{"Nota": n, "Quanti": R.GRADE[n][0] if n != "n/a" else "-",
                             "Quali": R.GRADE[n][1] if n != "n/a" else "-", "e(nota)": e[n], "MC": out["margem"][n],
                             "Multiplicador": 1 + out["margem"][n]} for n in R.NOTAS])
        st.dataframe(fmt(tab, pcts=["e(nota)", "MC"], nums=["Multiplicador"]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- premissa e RAS
with aba_ras:
    st.markdown("O **capital** cobre os modelos do escopo (behavior de PD em crédito; marcação a mercado de nível 2 "
                "e 3). A **RAS** mede a cobertura de validação do inventário total. Como premissa prudencial, todos "
                "os modelos do escopo devem estar com validação ativa e em dia; se não estiverem, recebem margem de "
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
                                  "Válida até", "Situação", "Exposição (R$ mi)", "Capital (R$ mi)"]]
        st.dataframe(fmt(datas(d), nums=["Exposição (R$ mi)", "Capital (R$ mi)"]), hide_index=True, width="stretch")
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
                "data-base com cada nota. O desvio medido não muda; muda a margem de conservadorismo.")
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
                "parâmetros da barra lateral: racional da modelagem por família, margem de conservadorismo, execução "
                "do cálculo por modelo, cenários, premissa e RAS, limitações e referências.")
    assinatura = (edit.to_csv(index=False), str(data_base), alfa, conf_s, int(tot_fora), int(val_fora), cob_min)
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
