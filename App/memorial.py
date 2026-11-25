"""Memorial de cálculo em formato de paper (PDF): racional da modelagem, execução sobre a base de entrada e referências."""
from __future__ import annotations

import datetime as dt
import io
import os
from dataclasses import replace

import numpy as np
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from scipy.stats import beta as B

import calculo as C
import rmcap as R

# ----------------------------------------------------------------------------- fontes e estilos
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_FONTES_OK = False


def _fontes():
    global _FONTES_OK
    if _FONTES_OK:
        return
    for nome, arq in (("Serif", "Regular"), ("Serif-B", "Bold"), ("Serif-I", "Italic"), ("Serif-BI", "BoldItalic")):
        pdfmetrics.registerFont(TTFont(nome, os.path.join(_DIR, f"LiberationSerif-{arq}.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
    _FONTES_OK = True


def _estilos():
    base = dict(fontName="Serif", fontSize=10.5, leading=14.2)
    return {
        "titulo": ParagraphStyle("titulo", fontName="Serif-B", fontSize=17, leading=21, alignment=TA_CENTER,
                                 spaceAfter=6),
        "subtitulo": ParagraphStyle("subtitulo", fontName="Serif-I", fontSize=11.5, leading=15, alignment=TA_CENTER,
                                    spaceAfter=4),
        "autor": ParagraphStyle("autor", fontName="Serif", fontSize=10.5, leading=14, alignment=TA_CENTER,
                                spaceAfter=14),
        "resumo": ParagraphStyle("resumo", fontName="Serif", fontSize=9.8, leading=13.2, alignment=TA_JUSTIFY,
                                 leftIndent=1.0 * cm, rightIndent=1.0 * cm, spaceAfter=6),
        "h1": ParagraphStyle("h1", fontName="Serif-B", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName="Serif-B", fontSize=11, leading=14, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", alignment=TA_JUSTIFY, spaceAfter=6, firstLineIndent=0.5 * cm, **base),
        "p0": ParagraphStyle("p0", alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "eqn": ParagraphStyle("eqn", fontName="Serif", fontSize=10.5, leading=13, alignment=2),
        "eq": ParagraphStyle("eq", fontName="Serif-I", fontSize=10.5, leading=15, alignment=TA_CENTER,
                             spaceBefore=3, spaceAfter=7),
        "cap": ParagraphStyle("cap", fontName="Serif", fontSize=9.2, leading=11.5, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=3),
        "nota": ParagraphStyle("nota", fontName="Serif", fontSize=8.6, leading=10.8, alignment=TA_LEFT,
                               spaceAfter=8),
        "cel": ParagraphStyle("cel", fontName="Serif", fontSize=8.4, leading=10, alignment=TA_LEFT),
        "celc": ParagraphStyle("celc", fontName="Serif-B", fontSize=8.4, leading=10, alignment=TA_CENTER),
        "ref": ParagraphStyle("ref", fontName="Serif", fontSize=9.4, leading=12, alignment=TA_JUSTIFY,
                              leftIndent=0.6 * cm, firstLineIndent=-0.6 * cm, spaceAfter=4),
        "item": ParagraphStyle("item", alignment=TA_JUSTIFY, spaceAfter=3, leftIndent=0.6 * cm,
                               firstLineIndent=-0.35 * cm, **base),
    }


# ----------------------------------------------------------------------------- utilidades
num, pct = C.num, C.pct


def _data(d):
    return d.strftime("%d/%m/%Y") if d else "-"


def _tabela(cab, linhas, larg, est, alinha_dir=None, total=False):
    """Tabela monocromática no estilo de periódico: filetes acima e abaixo do cabeçalho e no fim."""
    dados = [[Paragraph(f"<b>{c}</b>", est["celc"]) for c in cab]]
    for ln in linhas:
        dados.append([str(v) if j in (alinha_dir or []) else Paragraph(str(v), est["cel"])
                      for j, v in enumerate(ln)])
    t = Table(dados, colWidths=larg, repeatRows=1, hAlign="CENTER")
    st = [("FONTNAME", (0, 1), (-1, -1), "Serif"), ("FONTSIZE", (0, 1), (-1, -1), 8.4),
          ("LEADING", (0, 1), (-1, -1), 10), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    for j in (alinha_dir or []):
        st.append(("ALIGN", (j, 1), (j, -1), "RIGHT"))
    if total:
        st += [("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black), ("FONTNAME", (0, -1), (-1, -1), "Serif-B")]
    t.setStyle(TableStyle(st))
    return t


def _bloco_tabela(titulo, tab, est, nota=None):
    itens = [Paragraph(titulo, est["cap"]), tab]
    if nota:
        itens.append(Paragraph(nota, est["nota"]))
    else:
        itens.append(Spacer(1, 6))
    if len(getattr(tab, "_cellvalues", [])) > 26:     # tabela longa: quebra entre páginas, cabeçalho repetido
        itens[0].keepWithNext = 1
        return itens
    return KeepTogether(itens)


def _rodape(canvas, doc, titulo):
    canvas.saveState()
    canvas.setFont("Serif", 8.5)
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.4)
    canvas.line(2.4 * cm, 1.55 * cm, A4[0] - 2.4 * cm, 1.55 * cm)
    canvas.drawString(2.4 * cm, 1.1 * cm, titulo)
    canvas.drawRightString(A4[0] - 2.4 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------- figuras
def _fig_regua(cal, rg):
    ordem = sorted(cal, key=cal.get)
    d = Drawing(15.5 * cm, 6.4 * cm)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 1.4 * cm, 1.2 * cm, 13.6 * cm, 4.6 * cm
    bc.data = [[100 * cal[n] for n in ordem], [100 * rg["RC"][R.FAIXA[n]] for n in ordem]]
    bc.categoryAxis.categoryNames = ordem
    bc.categoryAxis.labels.fontName = "Serif"
    bc.categoryAxis.labels.fontSize = 8.5
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0, 26, 5
    bc.valueAxis.labels.fontName = "Serif"
    bc.valueAxis.labels.fontSize = 8
    bc.valueAxis.labelTextFormat = lambda v: f"{v:.0f}%"
    bc.bars[0].fillColor = colors.Color(0.25, 0.25, 0.25)
    bc.bars[1].fillColor = colors.Color(0.78, 0.78, 0.78)
    bc.bars.strokeColor = colors.black
    bc.bars.strokeWidth = 0.3
    bc.barSpacing, bc.groupSpacing = 1, 6
    d.add(bc)
    for i, (nome, cor) in enumerate((("FPR calibrado", bc.bars[0].fillColor), ("Régua padronizada", bc.bars[1].fillColor))):
        x = 2.0 * cm + i * 4.2 * cm
        from reportlab.graphics.shapes import Rect
        d.add(Rect(x, 6.0 * cm, 0.35 * cm, 0.25 * cm, fillColor=cor, strokeColor=colors.black, strokeWidth=0.3))
        d.add(String(x + 0.5 * cm, 6.03 * cm, nome, fontName="Serif", fontSize=8.5))
    d.add(String(0.1 * cm, 0.35 * cm, "Notas em ordem crescente de FPR calibrado (crédito, tier G)", fontName="Serif-I",
                 fontSize=8))
    return d


def _fig_cauda(p0, fb):
    c = R.cauda(p0, "RC")
    a, b = c["min"], c["max"]
    xs = np.linspace(a, b, 160)
    ys = B.pdf((xs - a) / (b - a), c["alpha"], c["beta"]) / (b - a) / 100
    d = Drawing(15.5 * cm, 6.2 * cm)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 1.4 * cm, 1.0 * cm, 13.6 * cm, 4.6 * cm
    lp.data = [list(zip(100 * xs, ys))]
    lp.lines[0].strokeColor = colors.black
    lp.lines[0].strokeWidth = 1.1
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax, lp.xValueAxis.valueStep = 0, 2.0, 0.25
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 0, float(ys.max()) * 1.45
    for ax in (lp.xValueAxis, lp.yValueAxis):
        ax.labels.fontName = "Serif"
        ax.labels.fontSize = 8
    lp.xValueAxis.labelTextFormat = lambda v: f"{v:.2f}".replace(".", ",")
    lp.yValueAxis.labelTextFormat = lambda v: f"{v:.1f}".replace(".", ",")
    d.add(lp)
    sx = lambda v: lp.x + (v - lp.xValueAxis.valueMin) / (lp.xValueAxis.valueMax - lp.xValueAxis.valueMin) * lp.width
    sy = lambda v: lp.y + (v - lp.yValueAxis.valueMin) / (lp.yValueAxis.valueMax - lp.yValueAxis.valueMin) * lp.height
    from reportlab.graphics.shapes import Circle
    pts = []
    for n in sorted(R.NOTAS, key=lambda k: fb[k]["fator"]):
        xv = 100 * fb[n]["fator"]
        yv = B.pdf((fb[n]["fator"] - a) / (b - a), c["alpha"], c["beta"]) / (b - a) / 100
        pts.append([n, sx(xv), sy(yv)])
    # rótulos numa linha acima da curva, afastados ao menos `gap` pontos, com linha de chamada até o ponto
    gap, xs_l = 13.0, [q[1] for q in pts]
    for _ in range(200):
        mudou = False
        for k in range(1, len(xs_l)):
            if xs_l[k] - xs_l[k - 1] < gap:
                m_ = (xs_l[k] + xs_l[k - 1]) / 2
                xs_l[k - 1], xs_l[k] = m_ - gap / 2, m_ + gap / 2
                mudou = True
        if not mudou:
            break
    from reportlab.graphics.shapes import Line
    y_rot = lp.y + lp.height + 6
    for (n, px, py), lx in zip(pts, xs_l):
        d.add(Line(px, py + 2.5, lx, y_rot - 2, strokeColor=colors.Color(0.55, 0.55, 0.55), strokeWidth=0.4,
                   strokeDashArray=[1.5, 1.5]))
        d.add(Circle(px, py, 2.2, fillColor=colors.white, strokeColor=colors.black, strokeWidth=0.8))
        d.add(String(lx, y_rot, n, fontName="Serif-B", fontSize=8, textAnchor="middle"))
    d.add(String(1.4 * cm, 0.2 * cm, "Fator de capital (% da exposição)", fontName="Serif-I", fontSize=8))
    return d




# ----------------------------------------------------------------------------- abordagem interna: figuras
def _fig_capital(det):
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    d2 = det.sort_values("Capital (R$ mi)")
    n = len(d2)
    alt = max(5.0, 0.42 * n + 1.6) * cm
    d = Drawing(15.5 * cm, alt)
    bc = HorizontalBarChart()
    bc.x, bc.y, bc.width, bc.height = 1.9 * cm, 1.0 * cm, 12.8 * cm, alt - 1.5 * cm
    bc.data = [list(d2["Capital (R$ mi)"])]
    bc.categoryAxis.categoryNames = list(d2["ID"])
    bc.categoryAxis.labels.fontName = "Serif"
    bc.categoryAxis.labels.fontSize = 7.5
    bc.valueAxis.valueMin = 0
    bc.valueAxis.labels.fontName = "Serif"
    bc.valueAxis.labels.fontSize = 7.5
    bc.valueAxis.labelTextFormat = lambda v: f"{v:,.0f}".replace(",", ".")
    tons = {"Crédito (behavior de PD)": 0.25, "Mercado nível 2": 0.55, "Mercado nível 3": 0.80}
    bc.bars.strokeColor = colors.black
    bc.bars.strokeWidth = 0.3
    for i, fam in enumerate(d2["Família"]):
        g = tons.get(fam, 0.5)
        bc.bars[(0, i)].fillColor = colors.Color(g, g, g)
    d.add(bc)
    from reportlab.graphics.shapes import Rect
    for j, (fam, g) in enumerate(tons.items()):
        x = 1.9 * cm + j * 4.4 * cm
        d.add(Rect(x, 0.15 * cm, 0.35 * cm, 0.25 * cm, fillColor=colors.Color(g, g, g), strokeColor=colors.black,
                   strokeWidth=0.3))
        d.add(String(x + 0.5 * cm, 0.18 * cm, fam, fontName="Serif", fontSize=8))
    return d


def _fig_margem(mc):
    ordem = sorted(mc, key=mc.get)
    d = Drawing(15.5 * cm, 5.6 * cm)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 1.4 * cm, 0.9 * cm, 13.6 * cm, 4.3 * cm
    bc.data = [[100 * mc[n] for n in ordem]]
    bc.categoryAxis.categoryNames = ordem
    bc.categoryAxis.labels.fontName = "Serif"
    bc.categoryAxis.labels.fontSize = 8.5
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0, 100, 20
    bc.valueAxis.labels.fontName = "Serif"
    bc.valueAxis.labels.fontSize = 8
    bc.valueAxis.labelTextFormat = lambda v: f"{v:.0f}%"
    bc.bars[0].fillColor = colors.Color(0.35, 0.35, 0.35)
    bc.bars.strokeColor = colors.black
    bc.bars.strokeWidth = 0.3
    d.add(bc)
    return d


def _v(x, f):
    return "-" if x is None or (isinstance(x, float) and x != x) else f(x)


# ----------------------------------------------------------------------------- abordagem interna: documento
def gerar_pdf_interno(p, inv: list, data_base: dt.date, out: dict, rs: dict, cen, atual: float, teto: float,
                      ref: float) -> bytes:
    import rmint as I
    import interno as N
    _fontes()
    est = _estilos()
    det, resumo = out["detalhe"], out["resumo"]
    mc, e = out["margem"], I.perda_relativa(p)
    hoje = dt.date.today()
    cont = {"t": 0, "f": 0}

    def nt():
        cont["t"] += 1
        return cont["t"]

    def nf():
        cont["f"] += 1
        return cont["f"]

    eqs = {}

    def eq(tex, chave=None):
        """Equação em destaque, centrada, com número à direita quando tem chave."""
        from formulas import desenho
        d_ = desenho(tex, 11.5)
        rot = ""
        if chave:
            eqs[chave] = len(eqs) + 1
            rot = f"({eqs[chave]})"
        larg = 16.2 * cm
        if d_.width > larg - 2.4 * cm:
            esc = (larg - 2.4 * cm) / d_.width
            d_.scale(esc, esc)
            d_.width, d_.height = d_.width * esc, d_.height * esc
        t_ = Table([["", d_, Paragraph(rot, est["eqn"])]], colWidths=[1.2 * cm, larg - 2.4 * cm, 1.2 * cm])
        t_.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return t_

    def ref_eq(chave):
        return str(eqs.get(chave, "?"))

    def mt(x):
        return num(x).replace(",", "{,}")

    def mp(x):
        return mt(100 * x) + r"\%"

    z, zc = p.z, float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(p.conf_sens))
    tot = resumo[resumo["Família"] == "Total"].iloc[0]
    s = []

    # ------------------------------------------------------------------ capa e resumo
    s.append(Spacer(1, 0.6 * cm))
    s.append(Paragraph("Capital econômico para risco de modelo", est["titulo"]))
    s.append(Paragraph("Memorial de cálculo pela abordagem interna de desvio medido", est["subtitulo"]))
    s.append(Paragraph(f"Data-base do cálculo: {_data(data_base)} · Documento gerado em {_data(hoje)}", est["autor"]))
    s.append(Paragraph(
        "<b>Resumo.</b> Este memorial documenta o cálculo do capital econômico de Pilar 2 para risco de modelo pela "
        f"abordagem interna, sobre {len(det)} modelos e exposição de R$ {num(tot['Exposição (R$ mi)'])} milhões. Para "
        "cada modelo, o capital é a exposição multiplicada pelo desvio do modelo no quantil de "
        f"{pct(p.alfa)} em um ano, medido em indicadores internos de performance, e por um fator de margem de "
        "conservadorismo definido pela nota da validação independente. Em crédito, o desvio vem da comparação, por "
        "grupo homogêneo (GH), entre a PD do GH e a BadRate observada no mesmo GH, descontado o ruído amostral, "
        "traduzida em falta de capital (fórmula IRB) ou de "
        "provisão, o que for maior. Em mercado de nível 2, vem da série do desvio da verificação independente de "
        "preços (IPV). Em nível 3, vem da sensibilidade do valor justo a premissas alternativas razoavelmente "
        "possíveis, divulgada pelo CPC 46, ou da diferença para um modelo alternativo. O horizonte do desvio é o "
        f"ciclo de validação do tier. O capital econômico requerido é de <b>R$ {num(out['capital'])} milhões</b>, "
        f"equivalente a {pct(tot['Taxa de capital'])} da exposição em escopo.", est["resumo"]))
    s.append(Paragraph("<b>Palavras-chave:</b> risco de modelo; Pilar 2; ICAAP; capital econômico; validação de "
                       "modelos; IPV; valor justo de nível 3; margem de conservadorismo.", est["resumo"]))

    # ------------------------------------------------------------------ 1. introdução
    s.append(Paragraph("1. Introdução", est["h1"]))
    s.append(Paragraph(
        "O risco de modelo é a possibilidade de perdas decorrentes de decisões baseadas em modelos incorretos ou mal "
        "utilizados (Derman, 1996; Federal Reserve e OCC, 2011). A Resolução CMN nº 4.557/2017 exige que a "
        "instituição avalie a suficiência de capital para os riscos a que está exposta, considerando o seu próprio "
        "perfil de risco. Uma régua geral, calibrada fora da instituição, é adequada quando o regulador define uma "
        "regra comum para bancos distintos; quando a instituição dispõe de indicadores de desempenho de seus "
        "próprios modelos, a regra geral deixa de usar a informação mais relevante sobre o erro de cada modelo.",
        est["p"]))
    s.append(Paragraph(
        "A abordagem interna mede, para cada modelo, quanto ele pode errar, com base na série de monitoramento, e "
        "traduz esse erro em perda potencial sobre a exposição que o modelo atende. A nota da validação "
        "independente, o tier e a vigência da validação continuam sendo os elementos de governança: a nota define "
        "o quanto o desvio medido é confiável, e o tier define por quanto tempo um erro pode permanecer sem "
        "detecção. A régua padronizada anterior é mantida apenas como referência de comparação.", est["p"]))

    # ------------------------------------------------------------------ 2. escopo
    s.append(Paragraph("2. Escopo e base de entrada", est["h1"]))
    s.append(Paragraph(
        "Em risco de crédito, o escopo considera um modelo de behavior (PD) por carteira, com exposição igual ao "
        "saldo contábil. Em risco de mercado, os modelos de marcação a mercado de instrumentos de nível 2 e 3 da "
        "hierarquia de valor justo (CPC 46), incluindo a contabilidade de hedge (Resolução CMN nº 4.966/2021), com "
        "exposição igual ao valor justo contábil absoluto.", est["p"]))
    linhas = [[r["ID"], r["Risco"], r["Carteira ou família"], r["Tier"], r["Nível"], r["Nota informada"],
               _data(r["Última validação"]), num(r["Exposição (R$ mi)"])] for _, r in det.iterrows()]
    s.append(_bloco_tabela(f"Tabela {nt()}. Base de entrada: modelos no escopo do capital (R$ milhões).", _tabela(
        ["ID", "Risco", "Carteira ou família", "Tier", "Nível", "Nota", "Última validação", "Exposição"], linhas,
        [1.3 * cm, 1.6 * cm, 3.6 * cm, 1.0 * cm, 1.1 * cm, 1.1 * cm, 2.4 * cm, 2.4 * cm], est, alinha_dir=[7]), est,
        "Tier G: validação anual. Tier M: validação a cada três anos. Nota em branco indica modelo sem validação."))

    # ------------------------------------------------------------------ 3. metodologia
    s.append(Paragraph("3. Metodologia", est["h1"]))
    s.append(Paragraph("3.1 Formato do cálculo", est["h2"]))
    s.append(Paragraph("Para cada modelo <i>i</i> do escopo, com exposição <i>E<sub>i</sub></i>, desvio "
                       "<i>D<sub>i</sub></i> no quantil <i>α</i> e no horizonte <i>h</i>, e margem de conservadorismo "
                       "<i>MC</i> dada pela nota efetiva da validação, o capital é:", est["p0"]))
    s.append(eq(r"$\mathrm{Capital}_i = E_i \cdot D_i(\alpha, h) \cdot \left[1 + MC(\mathrm{nota}_i)\right],"
                r"\qquad \mathrm{Capital} = \sum_{i=1}^{n} \mathrm{Capital}_i$", "cap"))
    s.append(Paragraph(
        f"O quantil adotado é <i>α</i> = {pct(p.alfa)}, o mesmo nível de confiança da fórmula IRB (BCBS, 2005), com "
        f"<i>z<sub>α</sub></i> = Φ<super>−1</super>(<i>α</i>) = {num(z)}. A soma em (" + ref_eq("cap") + ") não "
        "reconhece diversificação entre modelos, o que é conservador.", est["p0"]))
    s.append(Paragraph("3.2 Horizonte do desvio", est["h2"]))
    s.append(Paragraph(
        "Em crédito, um erro de calibragem permanece sem detecção, no máximo, até a validação seguinte. O horizonte "
        "<i>h</i> é o ciclo de validação do tier: <i>h</i> = 1 ano no tier G e <i>h</i> = 3 anos no tier M. "
        "Assume-se como premissa que o desvio evolui como passeio aleatório entre validações, de modo que sua "
        "dispersão cresce com √<i>h</i>. Em mercado, o horizonte é de um ano para os dois tiers: no nível 2, a "
        "verificação independente de preços é um controle mensal que ajusta a marcação quando o desvio passa da "
        "tolerância, de modo que o erro não se acumula até a validação seguinte; no nível 3, a sensibilidade "
        "divulgada já é uma faixa de premissas plausíveis, e não uma volatilidade que cresce no tempo.", est["p0"]))

    s.append(Paragraph("3.3 Crédito: desvio de calibragem por grupo homogêneo", est["h2"]))
    s.append(Paragraph(
        "No modelo de behavior, cada contrato recebe um grupo homogêneo (GH) de risco, e cada GH tem uma PD fixada na "
        "calibragem, próxima da BadRate observada naquela faixa de score no desenvolvimento. Se o cliente piora, ele "
        "migra para um GH de PD maior; essa migração é risco de crédito e já está capitalizada no Pilar 1. O risco de "
        "modelo é a distância entre a PD do GH e a BadRate que o mesmo GH apresenta no monitoramento. Para cada GH "
        "<i>g</i> e janela <i>t</i> de doze meses, com <i>n<sub>g,t</sub></i> contratos, define-se o desvio absoluto:",
        est["p0"]))
    s.append(eq(r"$d_{g,t} = BR_{g,t} - PD_g, \qquad BR_{g,t} = \dfrac{\mathrm{bads}_{g,t}}{n_{g,t}}$", "d"))
    s.append(Paragraph(
        "O desvio absoluto, e não a razão entre BadRate e PD, faz com que GHs de PD baixa gerem naturalmente desvios "
        "pequenos e dispensa o logaritmo quando a BadRate é zero. Só conta o desvio que não é ruído amostral, em dois "
        "componentes. O <b>viés</b> <i>b<sub>g</sub></i> é a diferença entre a BadRate acumulada da série e a PD do GH, "
        "considerada apenas se a PD do GH ficar fora do intervalo de Jeffreys de 95% da BadRate acumulada, que é o "
        "teste de calibragem por grau de rating do BCE (2019). Como as janelas de doze meses se sobrepõem, o tamanho "
        "amostral do teste é o número médio de contratos por janela:", est["p0"]))
    s.append(eq(r"$\overline{BR}_g = \dfrac{\sum_t \mathrm{bads}_{g,t}}{\sum_t n_{g,t}}, \qquad "
                r"b_g = \left(\overline{BR}_g - PD_g\right) \cdot \mathbf{1}\left[PD_g \notin IC^{J}_{95\%}"
                r"\left(\overline{BR}_g;\ \bar{n}_g\right)\right]$", "vies"))
    s.append(Paragraph(
        "A <b>volatilidade</b> <i>σ<sub>g</sub></i> é a variância observada do desvio menos a variância que o "
        "tamanho da amostra explica sozinho (binomial). Ela entra apenas se o teste de sobredispersão indicar "
        "variação além do ruído binomial, isto é, se a estatística (<i>k</i> − 1) <i>s</i><super>2</super> / "
        "<i>v<sub>g</sub></i> superar o quantil de 95% da distribuição χ<super>2</super> com <i>k</i> − 1 graus de "
        "liberdade, sendo <i>k</i> o número de janelas:", est["p0"]))
    s.append(eq(r"$v_g = \dfrac{1}{k}\sum_t \dfrac{PD_g\,(1 - PD_g)}{n_{g,t}}, \qquad "
                r"\sigma_g^2 = \max\left(0,\ s^2_{d_g} - v_g\right) \cdot \mathbf{1}\left["
                r"\dfrac{(k-1)\, s^2_{d_g}}{v_g} > \chi^2_{0{,}95;\,k-1}\right]$", "sig"))
    s.append(Paragraph("A PD do GH no quantil <i>α</i> e no horizonte <i>h</i> é:", est["p0"]))
    s.append(eq(r"$PD^{*}_g = PD_g + \max\left(0,\ b_g + z_{\alpha}\, \sigma_g \sqrt{h}\right)$", "pds"))
    s.append(Paragraph(
        "A PD estressada é traduzida em perda de duas formas, e adota-se a maior. A primeira é a falta de capital: a "
        "diferença entre o capital <i>K</i> da fórmula IRB (BCBS, 2005; Gordy, 2003; Vasicek, 2002) avaliado na "
        "PD estressada e na PD do GH, com correlação <i>R</i> e ajuste de prazo <i>MA</i> da "
        "classe (Tabela " + str(cont["t"] + 1) + "). A curva de <i>K</i> é íngreme nas PDs baixas, por isso o filtro "
        "de ruído é essencial: um desvio de 0,3% para 0,6% só gera capital se for estatisticamente real.",
        est["p0"]))
    s.append(eq(r"$K(PD) = LGD \left[\Phi\left(\dfrac{\Phi^{-1}(PD) + \sqrt{R}\,\Phi^{-1}(0{,}999)}{\sqrt{1-R}}"
                r"\right) - PD\right] \cdot MA$", "k"))
    s.append(Paragraph(
        "A segunda é a falta de provisão: a perda esperada adicional em doze meses que a PD subestimada deixaria de "
        "provisionar (Resolução CMN nº 4.966/2021). O desvio do modelo é a média dos desvios dos GHs ponderada pela "
        "participação <i>w<sub>g</sub></i> de cada GH no saldo:", est["p0"]))
    s.append(eq(r"$D_{RC} = \sum_g w_g \max\left\{K(PD^{*}_g) - K(PD_g);\ \ \left(PD^{*}_g - PD_g\right)"
                r" \cdot LGD\right\}$", "drc"))
    s.append(_bloco_tabela(f"Tabela {nt()}. Correlação de ativos e ajuste de prazo da fórmula IRB por classe.", _tabela(
        ["Classe", "Correlação R", "Ajuste de prazo MA"],
        [["Corporativo", "0,12 a 0,24, decrescente na PD", f"Sim (M = {num(p.maturidade)})"],
         ["Varejo imobiliário", "0,15", "Não (MA = 1)"], ["Varejo rotativo", "0,04", "Não (MA = 1)"],
         ["Varejo outros", "0,03 a 0,16, decrescente na PD", "Não (MA = 1)"]],
        [4.0 * cm, 5.4 * cm, 3.4 * cm], est), est))
    s.append(Paragraph(
        "A discriminação, medida pelo Gini, não é traduzida diretamente em perda: ela entra pela nota quantitativa "
        "da validação e, portanto, pela margem de conservadorismo (Seção 3.7). O teste de discriminação usado como "
        "referência é o de comparação da AUC atual com a do desenvolvimento (BCE, 2019; Engelmann, Hayden e "
        "Tasche, 2003).", est["p"]))

    s.append(Paragraph("3.4 Mercado de nível 2: desvio da verificação independente de preços", est["h2"]))
    s.append(Paragraph(
        "O indicador é o desvio <i>d</i> da verificação independente de preços (IPV): a diferença entre a marcação do "
        "modelo e uma fonte independente (taxas indicativas da Anbima, ajustes da B3, cotações e avaliação da "
        "contraparte), em % do valor justo, acompanhada mensalmente por família de instrumento <i>f</i> (BCBS, 2009; "
        "Resolução CMN nº 4.277/2013). A fonte de referência tem incerteza própria (spread entre compra e venda, "
        "dispersão entre fontes, largura do intervalo indicativo), resumida na tolerância <i>τ</i> da política de "
        "apreçamento. Errar dentro dessa faixa não é erro de modelo, e só conta o excesso sobre a tolerância:",
        est["p0"]))
    s.append(eq(r"$e_{f,t} = \mathrm{sinal}\left(d_{f,t}\right) \cdot \max\left(0,\ \left|d_{f,t}\right| - "
                r"\tau_{f,t}\right)$", "exc"))
    s.append(Paragraph(
        "O excesso médio é o viés da família, considerado apenas se o teste <i>t</i> rejeitar média zero a 95%, com "
        "<i>k</i> observações; a volatilidade é o desvio-padrão do excesso:", est["p0"]))
    s.append(eq(r"$\bar{e}_f = \dfrac{1}{k}\sum_t e_{f,t}, \qquad b_f = \bar{e}_f \cdot \mathbf{1}\left["
                r"\dfrac{\left|\bar{e}_f\right|}{s_{e_f} / \sqrt{k}} > t_{0{,}975;\,k-1}\right], \qquad "
                r"\sigma_f = s_{e_f}$", "bf"))
    s.append(Paragraph(
        "O desvio é bilateral, pois errar para cima ou para baixo é falha de apreçamento. O desvio da família é o "
        "quantil <i>α</i> do excesso absoluto, e o do modelo é a média ponderada pela participação <i>w<sub>f</sub></i> "
        "de cada família no valor justo:", est["p0"]))
    s.append(eq(r"$e_f \sim N\left(b_f,\ \sigma_f^2\right), \qquad \mathrm{Pr}\left(\left|e_f\right| \leq D_f\right)"
                r" = \alpha, \qquad D_{N2} = \sum_f w_f\, D_f$", "dn2"))

    s.append(Paragraph("3.5 Mercado de nível 3: sensibilidade a premissas não observáveis", est["h2"]))
    s.append(Paragraph(
        "No nível 3 não há preço observável para comparação. O indicador é a sensibilidade <i>s</i> do valor justo a "
        "premissas alternativas razoavelmente possíveis para os insumos não observáveis, que a instituição já "
        "calcula e divulga nas demonstrações financeiras (CPC 46, item 93(h)(ii)). Quando existe modelo "
        "alternativo, usa-se também a diferença <i>a</i> entre o modelo oficial e o alternativo, que é a medida de "
        "risco de modelo da literatura (Cont, 2006; Barrieu e Scandolo, 2015) e do ajuste de avaliação prudente "
        "(Regulamento Delegado (UE) 2016/101, art. 11). Atribui-se à sensibilidade divulgada o nível de confiança "
        f"<i>c</i> = {pct(p.conf_sens)}, o mesmo do ajuste de avaliação prudente, e leva-se o valor ao quantil "
        "<i>α</i>:", est["p0"]))
    s.append(eq(r"$D_{N3} = \max(s, a) \cdot \dfrac{z_{\alpha}}{z_{c}}$", "dn3"))
    s.append(Paragraph(
        f"Com os parâmetros adotados, <i>z<sub>α</sub></i> / <i>z<sub>c</sub></i> = {num(z)} / {num(zc)} = "
        f"{num(z / zc)}. O tratamento mantém o pior caso limitado ao conjunto de avaliações plausíveis, como na "
        "literatura de risco de modelo (Cont, 2006; Glasserman e Xu, 2014), em vez de considerar a exposição inteira "
        "em risco.", est["p0"]))

    s.append(Paragraph("3.6 Modelos sem indicador", est["h2"]))
    s.append(Paragraph(
        "Se um modelo não tem indicador de desvio, adota-se o maior desvio medido entre os modelos da mesma família. "
        "Se nenhum modelo da família tiver indicador, adota-se o fator de pior caso da régua padronizada de "
        "referência. Em ambos os casos, o modelo aparece sinalizado na execução.", est["p0"]))

    s.append(Paragraph("3.7 Margem de conservadorismo pela nota da validação", est["h2"]))
    s.append(Paragraph(
        "A nota da validação independente combina um eixo quantitativo e um qualitativo, com três níveis cada "
        "(Robusto, Satisfatório, Insatisfatório), resultando nas notas de A a I da matriz de cores. Quanto pior a "
        "nota, menos confiável é o desvio medido, e maior a margem de conservadorismo, no espírito das categorias "
        "de margem da EBA (2017). A perda relativa <i>e</i> combina, nos dois eixos, a probabilidade de falha "
        "<i>P</i> e a proporção de magnitude <i>π</i> das Tabelas 1 e 2 de Basileia (BCBS, 1996), com peso <i>ω</i> "
        "do eixo qualitativo, sendo <i>q</i> o nível quantitativo e <i>ql</i> o qualitativo da nota:", est["p0"]))
    s.append(eq(r"$e(\mathrm{nota}) = \dfrac{P_{q}\, \pi_{q} + \omega\, P_{ql}\, \pi_{ql}}{1 + \omega}, "
                r"\qquad e(\mathrm{n/a}) = 1$", "e"))
    s.append(Paragraph("A margem é a perda relativa normalizada entre a nota A, que recebe zero, e o modelo sem "
                       "validação, que recebe 100%:", est["p0"]))
    s.append(eq(r"$MC(\mathrm{nota}) = \dfrac{e(\mathrm{nota}) - e(A)}{1 - e(A)}$", "mc"))
    linhas = [[n, R.GRADE[n][0] if n != "n/a" else "-", R.GRADE[n][1] if n != "n/a" else "-", pct(e[n]), pct(mc[n]),
               num(1 + mc[n])] for n in sorted(mc, key=mc.get)]
    s.append(_bloco_tabela(f"Tabela {nt()}. Margem de conservadorismo por nota (ω = {num(p.omega)}).", _tabela(
        ["Nota", "Quanti", "Quali", "e(nota)", "MC", "Multiplicador"], linhas,
        [1.3 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm], est, alinha_dir=[3, 4, 5]), est,
        "Validação vencida (mais de 12 meses no tier G ou 36 no tier M) é tratada como ausência de validação. A "
        "ordem das margens reproduz a matriz de cores: "
        + ("atendida." if I.ordem_margem_ok(p) else "não atendida.")))
    s.append(KeepTogether([_fig_margem(mc), Paragraph(f"Figura {nf()}. Margem de conservadorismo por nota.",
                                                      est["cap"])]))

    # ------------------------------------------------------------------ 4. execução
    s.append(Paragraph("4. Execução do cálculo sobre a base de entrada", est["h1"]))
    s.append(Paragraph("4.1 Situação de validação na data-base", est["h2"]))
    s.append(Paragraph(f"Na data-base de {_data(data_base)}, {rs['escopo_validos']} dos {rs['escopo_total']} "
                       "modelos do escopo estão com validação vigente.", est["p0"]))
    linhas = [[r["ID"], r["Tier"], r["Nota informada"], _data(r["Última validação"]), _data(r["Válida até"]),
               r["Situação"], r["Nota efetiva"], pct(r["Margem MC"])] for _, r in det.iterrows()]
    s.append(_bloco_tabela(f"Tabela {nt()}. Situação da validação e margem por modelo.", _tabela(
        ["ID", "Tier", "Nota", "Última validação", "Válida até", "Situação", "Nota efetiva", "MC"], linhas,
        [1.3 * cm, 1.0 * cm, 1.1 * cm, 2.4 * cm, 2.2 * cm, 2.6 * cm, 1.8 * cm, 2.0 * cm], est, alinha_dir=[7]), est))

    s.append(Paragraph("4.2 Desvio por modelo", est["h2"]))
    rc = [m for m in inv if m.risco == "RC"]
    if rc:
        dd = det.set_index("ID")
        lin_gh = []
        for m in rc:
            dv = I.desvio(m, p)
            for g in (dv or {}).get("ghs", []):
                lin_gh.append([m.id, str(g["gh"]), pct(g["pd"]), f"{g['n_medio']:,.0f}".replace(",", "."), pct(g["br"]),
                               f"{pct(g['ic_lo'])} a {pct(g['ic_hi'])}", pct(g["vies"]), pct(g["sigma"]),
                               pct(g["pd_estressada"]), pct(g["D"]), pct(g["peso"])])
        if lin_gh:
            s.append(_bloco_tabela(f"Tabela {nt()}. Crédito: calibragem por GH.", _tabela(
                ["ID", "GH", "PD", "n médio", "BadRate", "IC Jeffreys 95%", "Viés b", "σ líquida", "PD*", "D<sub>g</sub>",
                 "Peso"], lin_gh,
                [1.15 * cm, 0.75 * cm, 1.3 * cm, 1.35 * cm, 1.35 * cm, 3.15 * cm, 1.3 * cm, 1.35 * cm, 1.4 * cm,
                 1.3 * cm, 1.3 * cm], est, alinha_dir=[2, 3, 4, 6, 7, 8, 9, 10]), est,
                "n médio: contratos por janela. BadRate: acumulada na série. Viés diferente de zero só quando a PD fica "
                "fora do intervalo de Jeffreys; σ líquida igual a zero quando o teste de sobredispersão não rejeita "
                "o ruído binomial."))
        linhas = [[m.id, m.ind.get("classe") or "-", _v(m.ind.get("lgd"), pct),
                   f"{dd.loc[m.id, 'Horizonte (anos)']:.0f}", _v(dd.loc[m.id, "PD média (crédito)"], pct),
                   _v(dd.loc[m.id, "PD estressada média"], pct), _v(dd.loc[m.id, "Desvio capital IRB"], pct),
                   _v(dd.loc[m.id, "Desvio provisão"], pct), pct(dd.loc[m.id, "Desvio D"])] for m in rc]
        s.append(_bloco_tabela(f"Tabela {nt()}. Crédito: desvio por modelo.", _tabela(
            ["ID", "Classe", "LGD", "h", "PD média", "PD* média", "Falta de capital", "Falta de provisão", "D"],
            linhas, [1.2 * cm, 2.6 * cm, 1.5 * cm, 0.8 * cm, 1.7 * cm, 1.7 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm], est,
            alinha_dir=[2, 3, 4, 5, 6, 7, 8]), est, "Médias ponderadas pela participação de cada GH no saldo."))
    n2 = [m for m in inv if m.risco == "RM" and m.nivel == "N2"]
    if n2:
        dd = det.set_index("ID")
        lin_f = []
        for m in n2:
            dv = I.desvio(m, p)
            for f_ in (dv or {}).get("familias", []):
                lin_f.append([m.id, f_["familia"], pct(f_["d_medio"]), pct(f_["d_sd"]), pct(f_["tau"]),
                              pct(f_["fora_tol"]), pct(f_["e_medio"]), num(f_["t"]), pct(f_["vies"]),
                              pct(f_["sigma"]), pct(f_["D"]), pct(f_["peso"])])
        if lin_f:
            s.append(_bloco_tabela(f"Tabela {nt()}. Mercado de nível 2: IPV por família de instrumento (% do valor "
                                   "justo).", _tabela(
                ["ID", "Família", "d médio", "d d.p.", "τ", "Fora de τ", "e médio", "t", "Viés b", "σ", "D<sub>f</sub>",
                 "Peso"], lin_f,
                [1.15 * cm, 2.6 * cm, 1.2 * cm, 1.2 * cm, 1.1 * cm, 1.25 * cm, 1.2 * cm, 1.0 * cm, 1.2 * cm, 1.1 * cm,
                 1.2 * cm, 1.3 * cm], est, alinha_dir=list(range(2, 12))), est,
                "τ: tolerância média da fonte de referência. Fora de τ: fração das observações com desvio acima da "
                "tolerância. Viés diferente de zero só quando o teste t rejeita média zero a 95%, isto é, |t| acima "
                "do quantil de 97,5% da distribuição t com k − 1 graus de liberdade."))
        linhas = [[m.id, m.carteira, str(len(m.ind.get("fams") or [])), pct(dd.loc[m.id, "Desvio D"]),
                   dd.loc[m.id, "Fonte do desvio"]] for m in n2]
        s.append(_bloco_tabela(f"Tabela {nt()}. Mercado de nível 2: desvio por modelo.", _tabela(
            ["ID", "Carteira", "Famílias", "D", "Fonte"], linhas,
            [1.3 * cm, 3.6 * cm, 1.8 * cm, 2.0 * cm, 4.2 * cm], est, alinha_dir=[2, 3]), est,
            "D: média dos desvios das famílias ponderada pela participação no valor justo."))
    n3 = [m for m in inv if m.risco == "RM" and m.nivel != "N2"]
    if n3:
        dd = det.set_index("ID")
        linhas = [[m.id, m.carteira, _v(m.ind.get("sens"), pct), _v(m.ind.get("alt"), pct), num(z / zc),
                   num(dd.loc[m.id, "Horizonte (anos)"]), pct(dd.loc[m.id, "Desvio D"]), dd.loc[m.id, "Critério"]]
                  for m in n3]
        s.append(_bloco_tabela(f"Tabela {nt()}. Mercado de nível 3: sensibilidade e modelo alternativo.", _tabela(
            ["ID", "Família", "s", "a", "Escala", "h", "D", "Critério"], linhas,
            [1.2 * cm, 2.6 * cm, 1.5 * cm, 1.5 * cm, 1.4 * cm, 0.9 * cm, 1.8 * cm, 3.6 * cm], est,
            alinha_dir=[2, 3, 4, 5, 6]), est, "Escala = z<sub>α</sub> / z<sub>c</sub>."))

    s.append(Paragraph("4.3 Capital por modelo", est["h2"]))
    linhas = [[r["ID"], r["Carteira ou família"], r["Nota efetiva"], pct(r["Desvio D"]), pct(r["Margem MC"]),
               pct(r["Taxa de capital"]), num(r["Exposição (R$ mi)"]), num(r["Capital (R$ mi)"])]
              for _, r in det.iterrows()]
    linhas.append(["Total", "", "", "", "", pct(tot["Taxa de capital"]), num(tot["Exposição (R$ mi)"]),
                   num(tot["Capital (R$ mi)"])])
    s.append(_bloco_tabela(f"Tabela {nt()}. Capital interno por modelo (R$ milhões).", _tabela(
        ["ID", "Carteira ou família", "Nota", "D", "MC", "Taxa", "Exposição", "Capital"], linhas,
        [1.3 * cm, 3.5 * cm, 1.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.3 * cm, 2.0 * cm], est,
        alinha_dir=[3, 4, 5, 6, 7], total=True), est,
        "Taxa = D × (1 + MC). Capital = taxa × exposição. No total, a taxa é a média ponderada pela exposição."))
    if len(det):
        s.append(KeepTogether([_fig_capital(det), Paragraph(f"Figura {nf()}. Capital interno por modelo.",
                                                            est["cap"])]))

    s.append(Paragraph("4.4 Exemplos passo a passo", est["h2"]))
    for fam in ("RC", "N2", "N3"):
        sub = det[det["Família"] == I.FAMILIAS[fam]]
        if not len(sub):
            continue
        r = sub.loc[sub["Capital (R$ mi)"].idxmax()]
        m = next(x for x in inv if x.id == r["ID"])
        fam_txt = r["Família"][0].lower() + r["Família"][1:]
        s.append(Paragraph(
            f"<b>{r['ID']}, {r['Carteira ou família']}</b> ({fam_txt}, tier {r['Tier']}, <i>h</i> = "
            f"{r['Horizonte (anos)']:.0f} {'ano' if r['Horizonte (anos)'] == 1 else 'anos'}, nota efetiva {r['Nota efetiva']}, exposição de R$ "
            f"{num(r['Exposição (R$ mi)'])} milhões).", est["p0"]))
        h_ = f"{r['Horizonte (anos)']:.0f}"
        if fam == "RC" and r["Fonte do desvio"] == "Medido":
            dv = I.desvio(m, p)
            g = max(dv["ghs"], key=lambda x: x["peso"] * x["D"])
            s.append(Paragraph(f"No GH {g['gh']}, que mais contribui para o desvio do modelo (peso de "
                               f"{pct(g['peso'])} no saldo):", est["p0"]))
            s.append(eq(rf"$PD^{{*}}_{{{g['gh']}}} = {mp(g['pd'])} + \max\left(0,\ {mp(g['vies'])} + {mt(z)} \times "
                        rf"{mp(g['sigma'])} \times \sqrt{{{h_}}}\right) = {mp(g['pd_estressada'])}$"))
            s.append(eq(rf"$D_{{{g['gh']}}} = \max\left\{{{mp(g['D_K'])};\ \ {mp(g['D_PE'])}\right\}} = {mp(g['D'])}$"))
            if len(dv["ghs"]) <= 3:
                soma = r" + ".join(rf"{mt(x['peso'])} \times {mp(x['D'])}" for x in dv["ghs"])
                s.append(eq(rf"$D_{{RC}} = {soma} = {mp(dv['D'])}$"))
            else:
                s.append(Paragraph("Somando os GHs com os pesos e desvios da tabela de calibragem por GH:", est["p0"]))
                s.append(eq(rf"$D_{{RC}} = \sum_{{g=1}}^{{{len(dv['ghs'])}}} w_g\, D_g = {mp(dv['D'])}$"))
        elif fam == "N2" and r["Fonte do desvio"] == "Medido":
            dv = I.desvio(m, p)
            f_ = max(dv["familias"], key=lambda x: x["peso"] * x["D"])
            s.append(Paragraph(f"Na família {f_['familia']}, que mais contribui para o desvio do modelo (peso de "
                               f"{pct(f_['peso'])} no valor justo), com tolerância média de {pct(f_['tau'])}:",
                               est["p0"]))
            s.append(eq(rf"$e_f \sim N\left({mp(f_['vies'])},\ \left({mp(f_['sigma'])}\right)^2\right)\ \Rightarrow\ "
                        rf"D_f = {mp(f_['D'])}$"))
            soma = r" + ".join(rf"{mt(x['peso'])} \times {mp(x['D'])}" for x in dv["familias"])
            s.append(eq(rf"$D_{{N2}} = {soma} = {mp(dv['D'])}$"))
        elif fam == "N3" and r["Fonte do desvio"] == "Medido":
            sv, av = m.ind.get("sens"), m.ind.get("alt")
            base_ = (rf"\max\left({mp(sv)},\ {mp(av)}\right)" if (sv is not None and av is not None)
                     else mp(sv if sv is not None else av))
            s.append(eq(rf"$D_{{N3}} = {base_} \times {mt(z / zc)} = {mp(r['Desvio D'])}$"))
        else:
            s.append(Paragraph(f"Sem indicador; <i>D</i> = {pct(r['Desvio D'])} ({r['Fonte do desvio'].lower()}).",
                               est["p0"]))
        s.append(eq(rf"$\mathrm{{Capital}} = {mt(r['Exposição (R$ mi)'])} \times {mp(r['Desvio D'])} \times "
                    rf"\left(1 + {mp(r['Margem MC'])}\right) = \mathbf{{{mt(r['Capital (R$ mi)'])}}}$"))

    s.append(Paragraph("4.5 Consolidação", est["h2"]))
    linhas = [[r["Família"], str(r["Modelos"]), str(r["Sem validação vigente"]), num(r["Exposição (R$ mi)"]),
               pct(r["Taxa de capital"]), num(r["Capital (R$ mi)"])] for _, r in resumo.iterrows()]
    s.append(_bloco_tabela(f"Tabela {nt()}. Capital interno consolidado por família (R$ milhões).", _tabela(
        ["Família", "Modelos", "Sem validação", "Exposição", "Taxa", "Capital"], linhas,
        [4.4 * cm, 1.6 * cm, 2.2 * cm, 2.8 * cm, 2.0 * cm, 2.4 * cm], est, alinha_dir=[1, 2, 3, 4, 5], total=True),
        est))
    s.append(Paragraph(
        f"O capital econômico requerido para risco de modelo é de <b>R$ {num(out['capital'])} milhões</b>. Modelos sem "
        f"validação vigente respondem por {pct(rs['parcela_capital_descumprimento'])} do capital. Como referência, a "
        f"régua padronizada anterior resultaria em R$ {num(ref)} milhões"
        + (f"; a abordagem interna corresponde a {num(out['capital'] / ref)} vezes a referência." if ref else "."),
        est["p"]))

    s.append(Paragraph("4.6 Cenários", est["h2"]))
    s.append(Paragraph("O desvio medido não depende da nota; a nota altera a margem. A tabela mostra o capital se "
                       "todos os modelos, ou apenas os pendentes, fossem validados na data-base com cada nota.",
                       est["p0"]))
    linhas = [[r["Nota"], num(r["Todos validados com a nota"]), num(r["Só os pendentes validados com a nota"])]
              for _, r in cen.iterrows()]
    linhas += [["Nenhum validado", num(teto), num(teto)], ["Base de entrada", num(atual), num(atual)]]
    s.append(_bloco_tabela(f"Tabela {nt()}. Capital sob cenários de validação (R$ milhões).", _tabela(
        ["Nota", "Todos validados com a nota", "Só os pendentes validados"], linhas, [3.4 * cm, 4.4 * cm, 4.4 * cm],
        est, alinha_dir=[1, 2]), est))

    # ------------------------------------------------------------------ 5. premissa e RAS
    s.append(Paragraph("5. Premissa prudencial do escopo e apetite a riscos", est["h1"]))
    s.append(Paragraph(
        "O capital de risco de modelo e a declaração de apetite a riscos (RAS) medem populações diferentes. O capital "
        "cobre os modelos do escopo definido na Seção 2; a RAS mede a cobertura de validação do inventário total, "
        "com um mínimo definido pela instituição. Como premissa prudencial, todos os modelos do escopo devem estar "
        "com validação ativa e em dia. Se não estiverem, recebem margem de 100%, que funciona como penalidade por "
        "descumprimento da premissa.", est["p"]))
    linhas = [["Modelos no escopo do capital", str(rs["escopo_total"])],
              ["Com validação vigente no escopo", str(rs["escopo_validos"])],
              ["Premissa prudencial", "Atendida" if rs["premissa_ok"] else "Não atendida"],
              ["Capital de modelos fora da premissa (R$ mi)",
               f"{num(rs['capital_descumprimento'])} ({pct(rs['parcela_capital_descumprimento'])})"],
              ["Modelos fora do escopo do capital", str(rs["fora_total"])],
              ["Validados em dia fora do escopo", str(rs["fora_validos"])],
              ["Cobertura de validação do inventário total", pct(rs["cobertura"])],
              ["Mínimo da RAS", pct(rs["cobertura_min"])],
              ["Situação na RAS", "Dentro do apetite" if rs["ras_ok"] else "Fora do apetite"]]
    s.append(_bloco_tabela(f"Tabela {nt()}. Premissa do escopo e cobertura da RAS.", _tabela(
        ["Indicador", "Valor"], linhas, [8.0 * cm, 5.0 * cm], est, alinha_dir=[1]), est))

    # ------------------------------------------------------------------ 6. governança e limitações
    s.append(Paragraph("6. Governança e limitações", est["h1"]))
    s.append(Paragraph(
        "Os indicadores de desvio são produzidos pelo monitoramento (primeira linha) e verificados pela validação "
        "independente (segunda linha). Na revisão anual, reavaliam-se o nível de confiança, a confiança implícita "
        "da sensibilidade de nível 3, as classes IRB, a margem de conservadorismo e a coerência entre o desvio "
        "medido e a nota da validação. As principais limitações, incorporadas como premissas, são:", est["p"]))
    for t in ("no IPV, o excesso sobre a tolerância é tratado como normal, embora tenha massa em zero quando o "
              "desvio fica dentro da faixa; a tolerância de cada família deve refletir a incerteza real da fonte;",
              "o horizonte de um ano em mercado supõe que o IPV mensal corrige a marcação; famílias sem IPV regular "
              "devem ter o horizonte revisto;",
              "a BadRate de cada GH é tratada como binomial e os desvios de calibragem como normais; "
              "séries curtas tornam "
              "o quantil de 99,9% dependente dessa hipótese;",
              "janelas de doze meses sobrepostas no monitoramento de crédito: o teste de Jeffreys usa o número médio "
              "de contratos por janela, e a volatilidade medida pode ficar subestimada;",
              "GHs com poucos contratos tendem a não rejeitar o ruído binomial, e o desvio desses GHs fica igual a "
              "zero; a margem de conservadorismo da nota é o que cobre essa incerteza;",
              "em crédito, o crescimento do desvio com √h supõe passeio aleatório entre validações;",
              "a sensibilidade do nível 3 é tratada como equivalente a um quantil de "
              f"{pct(p.conf_sens)}; a convenção de cada mesa para premissas razoavelmente possíveis deve ser "
              "documentada;",
              "a soma entre modelos não reconhece diversificação nem correlação entre erros de modelo;",
              "a discriminação do modelo de crédito entra apenas pela nota, sem tradução direta em perda."):
        s.append(Paragraph("• " + t, est["item"]))

    # ------------------------------------------------------------------ referências
    s.append(Paragraph("Referências", est["h1"]))
    refs = [
        "BANCO CENTRAL EUROPEU. <i>Instructions for reporting the validation results of internal models: IRB "
        "Pillar I models for credit risk</i>. Frankfurt: ECB, 2019.",
        "BARRIEU, P.; SCANDOLO, G. Assessing financial model risk. <i>European Journal of Operational Research</i>, "
        "v. 242, n. 2, p. 546-556, 2015.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>Supervisory framework for the use of \"backtesting\" in "
        "conjunction with the internal models approach to market risk capital requirements</i>. Basel: BIS, 1996.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>An explanatory note on the Basel II IRB risk weight functions</i>. "
        "Basel: BIS, 2005.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>Supervisory guidance for assessing banks' financial instrument "
        "fair value practices</i>. Basel: BIS, 2009.",
        "COMITÊ DE PRONUNCIAMENTOS CONTÁBEIS. <i>CPC 46: Mensuração do valor justo</i>. Brasília, 2012.",
        "CONSELHO MONETÁRIO NACIONAL. Resolução CMN nº 4.277, de 31 de outubro de 2013. Dispõe sobre requisitos "
        "mínimos e ajustes prudenciais a serem observados no processo de apreçamento de instrumentos financeiros "
        "avaliados pelo valor de mercado.",
        "CONSELHO MONETÁRIO NACIONAL. Resolução CMN nº 4.557, de 23 de fevereiro de 2017. Dispõe sobre a estrutura de "
        "gerenciamento de riscos e de capital.",
        "CONSELHO MONETÁRIO NACIONAL. Resolução CMN nº 4.966, de 25 de novembro de 2021. Dispõe sobre conceitos e "
        "critérios contábeis aplicáveis a instrumentos financeiros e à contabilidade de hedge.",
        "CONT, R. Model uncertainty and its impact on the pricing of derivative instruments. <i>Mathematical "
        "Finance</i>, v. 16, n. 3, p. 519-547, 2006.",
        "DERMAN, E. <i>Model risk</i>. Goldman Sachs Quantitative Strategies Research Notes. New York, 1996.",
        "ENGELMANN, B.; HAYDEN, E.; TASCHE, D. Testing rating accuracy. <i>Risk</i>, v. 16, n. 1, p. 82-86, 2003.",
        "EUROPEAN BANKING AUTHORITY. <i>Guidelines on PD estimation, LGD estimation and the treatment of defaulted "
        "exposures</i> (EBA/GL/2017/16). London, 2017.",
        "EUROPEAN COMMISSION. Commission Delegated Regulation (EU) 2016/101 supplementing Regulation (EU) No 575/2013 "
        "with regard to regulatory technical standards for prudent valuation. <i>Official Journal of the European "
        "Union</i>, 2016.",
        "FEDERAL RESERVE BOARD; OFFICE OF THE COMPTROLLER OF THE CURRENCY. <i>SR 11-7: Supervisory guidance on model "
        "risk management</i>. Washington, 2011.",
        "GLASSERMAN, P.; XU, X. Robust risk measurement and model risk. <i>Quantitative Finance</i>, v. 14, n. 1, "
        "p. 29-58, 2014.",
        "GORDY, M. B. A risk-factor model foundation for ratings-based bank capital rules. <i>Journal of Financial "
        "Intermediation</i>, v. 12, n. 3, p. 199-232, 2003.",
        "KERKHOF, J.; MELENBERG, B.; SCHUMACHER, H. Model risk and capital reserves. <i>Journal of Banking &amp; "
        "Finance</i>, v. 34, n. 1, p. 267-279, 2010.",
        "VASICEK, O. The distribution of loan portfolio value. <i>Risk</i>, v. 15, n. 12, p. 160-162, 2002.",
    ]
    for r_ in refs:
        s.append(Paragraph(r_, est["ref"]))

    s = [x for item in s for x in (item if isinstance(item, list) else [item])]
    buf = io.BytesIO()
    titulo = "Capital econômico para risco de modelo: memorial de cálculo"
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.4 * cm, rightMargin=2.4 * cm, topMargin=2.2 * cm,
                            bottomMargin=2.2 * cm, title=titulo, author="Memorial gerado pela ferramenta de capital "
                            "de risco de modelo", subject="Capital econômico de Pilar 2 para risco de modelo")
    doc.build(s, onFirstPage=lambda c, d: _rodape(c, d, titulo), onLaterPages=lambda c, d: _rodape(c, d, titulo))
    return buf.getvalue()
