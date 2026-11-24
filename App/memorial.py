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
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = 0, float(ys.max()) * 1.25
    for ax in (lp.xValueAxis, lp.yValueAxis):
        ax.labels.fontName = "Serif"
        ax.labels.fontSize = 8
    lp.xValueAxis.labelTextFormat = lambda v: f"{v:.2f}".replace(".", ",")
    lp.yValueAxis.labelTextFormat = lambda v: f"{v:.1f}".replace(".", ",")
    d.add(lp)
    sx = lambda v: lp.x + (v - lp.xValueAxis.valueMin) / (lp.xValueAxis.valueMax - lp.xValueAxis.valueMin) * lp.width
    sy = lambda v: lp.y + (v - lp.yValueAxis.valueMin) / (lp.yValueAxis.valueMax - lp.yValueAxis.valueMin) * lp.height
    from reportlab.graphics.shapes import Circle
    ultimo_x = -99.0
    for n in sorted(R.NOTAS, key=lambda k: fb[k]["fator"]):
        xv = 100 * fb[n]["fator"]
        yv = B.pdf((fb[n]["fator"] - a) / (b - a), c["alpha"], c["beta"]) / (b - a) / 100
        d.add(Circle(sx(xv), sy(yv), 2.2, fillColor=colors.white, strokeColor=colors.black, strokeWidth=0.8))
        dy = 7 if sx(xv) - ultimo_x > 12 else 17
        d.add(String(sx(xv) - 3, sy(yv) + dy, n, fontName="Serif-B", fontSize=8))
        ultimo_x = sx(xv)
    d.add(String(1.4 * cm, 0.2 * cm, "Fator de capital (% da exposição)", fontName="Serif-I", fontSize=8))
    return d


# ----------------------------------------------------------------------------- documento
def gerar_pdf(p: R.Parametros, inv: list, data_base: dt.date, out: dict, rs: dict, cen, atual: float,
              teto: float) -> bytes:
    _fontes()
    est = _estilos()
    det, resumo, bruto = out["detalhe"], out["resumo"], out["bruto"]
    p0 = R.Parametros()
    pc = replace(p0, abordagem="calibrada")
    P, S, pi = p0.P, p0.S, R.PROPORCAO_BASILEIA
    sev = R.severidade_base(p0, "RC")
    fb = R.fator_base(p0, "RC")
    cal = {r: R.fpr_calibrado(pc, r) for r in R.RISCOS}
    rg = R.regua(p0)
    lo, hi = R.intervalo_k(p0)
    ciclo = R.lambda_ciclo(p0)
    salv = R.salvaguarda_ordenacao(p)
    cd = R.cauda(p0, "RC")
    k1, k2 = R.k_irb(0.01), R.k_irb(0.02)
    lm, ln2 = p.lam_tier["M"], p.lam_nivel["N2"]
    tot = resumo[resumo["Risco"] == "Total"].iloc[0]
    por = {r: resumo[resumo["Risco"] == r].iloc[0] for r in ("Crédito", "Mercado") if (resumo["Risco"] == r).any()}
    hoje = dt.date.today()
    n_mod = len(det)
    s = []

    # ------------------------------------------------------------------ capa e resumo
    s.append(Spacer(1, 0.6 * cm))
    s.append(Paragraph("Capital econômico para risco de modelo", est["titulo"]))
    s.append(Paragraph("Memorial de cálculo pela régua padronizada de FPR de risco de modelo", est["subtitulo"]))
    s.append(Paragraph(f"Data-base do cálculo: {_data(data_base)} · Documento gerado em {_data(hoje)}", est["autor"]))
    resumo_txt = (
        "<b>Resumo.</b> Este memorial documenta o cálculo do capital econômico de Pilar 2 para risco de modelo sobre "
        f"a base de entrada informada, com {n_mod} modelos e exposição total de R$ {num(tot['Exposição (R$ mi)'])} "
        "milhões. A metodologia segue o formato da abordagem padronizada de Basileia: cada modelo recebe um fator de "
        "ponderação de risco de modelo (FPR), definido pela cor da nota de validação em uma régua de sete faixas, "
        "que multiplica a exposição da carteira atendida pelo modelo; o capital é 8% do RWA resultante. Os valores "
        "da régua derivam de uma calibração em que a probabilidade e a magnitude de uma falha de modelo são tiradas "
        "das réguas de backtesting do Comitê de Basileia (1996) e da fórmula IRB, combinadas em dois canais de erro "
        "(quantitativo e qualitativo) e posicionadas em uma cauda de perda Beta-PERT. Para a base de entrada, o RWA "
        f"de risco de modelo é de <b>R$ {num(bruto['rwa'])} milhões</b>, com FPR médio de {pct(bruto['fpr_medio'])}, e o "
        f"capital econômico requerido é de <b>R$ {num(bruto['addon'])} milhões</b>, equivalente a "
        f"{pct(bruto['addon'] / bruto['saldo'] if bruto['saldo'] else 0)} da exposição em escopo.")
    s.append(Paragraph(resumo_txt, est["resumo"]))
    s.append(Paragraph("<b>Palavras-chave:</b> risco de modelo; Pilar 2; ICAAP; capital econômico; abordagem "
                       "padronizada; backtesting; distribuição Beta-PERT; validação de modelos.", est["resumo"]))

    # ------------------------------------------------------------------ 1. introdução
    s.append(Paragraph("1. Introdução", est["h1"]))
    s.append(Paragraph(
        "O risco de modelo é a possibilidade de perdas decorrentes de decisões baseadas em modelos incorretos ou "
        "mal utilizados (Derman, 1996; Federal Reserve e OCC, 2011). A Resolução CMN nº 4.557/2017 exige que a "
        "instituição identifique as limitações de seus modelos, submeta-os a avaliação independente e avalie a "
        "suficiência de capital para os riscos incorridos no processo de avaliação de adequação de capital (ICAAP). "
        "Como não há requerimento de Pilar 1 específico para risco de modelo, o tratamento em capital é uma decisão "
        "de Pilar 2.", est["p"]))
    s.append(Paragraph(
        "O impacto de um erro de modelo cresce com o tamanho da carteira que o modelo atende. Por isso, a "
        "metodologia adota o formato da abordagem padronizada (BCBS, 2004; Resolução BCB nº 229/2022): poucas "
        "categorias de risco, cada uma com um fator de ponderação em valor redondo, escolhido de forma conservadora "
        "a partir de uma calibração mais fina. Este memorial apresenta o escopo e a base de entrada (Seção 2), o "
        "racional da modelagem (Seção 3), a execução do cálculo (Seção 4), a premissa prudencial do escopo e a relação "
        "com a declaração de apetite a riscos (Seção 5), governança e limitações (Seção 6) e referências.", est["p"]))

    # ------------------------------------------------------------------ 2. escopo e base
    s.append(Paragraph("2. Escopo e base de entrada", est["h1"]))
    s.append(Paragraph(
        "O escopo do capital é a fatia material do inventário de modelos. Em risco de crédito, um modelo de "
        "behavior de probabilidade de inadimplência (PD) por carteira, com exposição igual ao saldo contábil da "
        "carteira. Em risco de mercado, os modelos de marcação a mercado de instrumentos classificados nos níveis 2 "
        "e 3 da hierarquia de valor justo (CPC 46), incluindo a contabilidade de hedge (Resolução CMN nº "
        "4.966/2021), com exposição igual ao valor justo contábil absoluto das posições. Instrumentos de nível 1 "
        "não dependem de modelo e ficam fora do escopo.", est["p"]))
    linhas = []
    for _, r in det.iterrows():
        linhas.append([r["ID"], r["Risco"], r["Carteira ou família"], r["Tier"], r["Nível"], r["Nota informada"],
                       _data(r["Última validação"]), num(r["Exposição (R$ mi)"])])
    s.append(_bloco_tabela(
        "Tabela 1. Base de entrada: modelos no escopo do capital (exposição em R$ milhões).",
        _tabela(["ID", "Risco", "Carteira ou família", "Tier", "Nível", "Nota", "Última validação", "Exposição"],
                linhas, [1.3 * cm, 1.6 * cm, 3.6 * cm, 1.0 * cm, 1.1 * cm, 1.1 * cm, 2.4 * cm, 2.4 * cm], est,
                alinha_dir=[7]),
        est, "Tier G: validação anual. Tier M: validação a cada três anos. Nível: hierarquia de valor justo, "
             "aplicável apenas a mercado. Nota em branco indica modelo sem validação."))

    # ------------------------------------------------------------------ 3. metodologia
    s.append(Paragraph("3. Metodologia", est["h1"]))
    s.append(Paragraph("3.1 Formato do cálculo", est["h2"]))
    s.append(Paragraph(
        "Cada modelo <i>i</i> recebe um FPR de risco de modelo igual ao valor da faixa de sua nota de validação, "
        "ajustado por fatores de tier e de nível de valor justo. O RWA de risco de modelo é a soma do FPR vezes a "
        "exposição, e o capital é 8% desse RWA:", est["p0"]))
    s.append(Paragraph("FPR<sub>i</sub> = λ<sub>tier</sub> × λ<sub>nível</sub> × FPR<sub>r</sub>(faixa da nota<sub>i</sub>),"
                       "   RWA<sub>RM</sub> = Σ<sub>i</sub> FPR<sub>i</sub> × E<sub>i</sub>,   "
                       "Capital = 8% × RWA<sub>RM</sub>", est["eq"]))
    s.append(Paragraph(
        f"com λ<sub>G</sub> = λ<sub>N3</sub> = 1, λ<sub>M</sub> = {num(lm)} e λ<sub>N2</sub> = {num(ln2)}; em crédito, "
        "λ<sub>nível</sub> = 1. O FPR mede apenas o risco de modelo: não há fator regulatório de crédito multiplicando "
        "a exposição, pois o risco de crédito da carteira já é capitalizado no Pilar 1.", est["p0"]))

    s.append(Paragraph("3.2 Nota de validação e matriz de cores", est["h2"]))
    s.append(Paragraph(
        "A validação independente classifica cada modelo em Robusto, Satisfatório ou Insatisfatório em dois eixos: "
        "quantitativo (discriminação, calibração, estabilidade, backtesting) e qualitativo (governança, dados, "
        "documentação, uso e monitoramento). A combinação define a nota de A a I. A matriz de cores de referência "
        "agrupa as notas em verde (A, B, D), amarelo (C, E), laranja (G) e vermelho (F, H, I). Validação vencida, "
        "isto é, com mais de 12 meses no tier G ou de 36 meses no tier M na data-base, é tratada como ausência de "
        "validação.", est["p"]))
    mat = [["Robusto", "C (amarelo)", "B (verde)", "A (verde)"],
           ["Satisfatório", "F (vermelho)", "E (amarelo)", "D (verde)"],
           ["Insatisfatório", "I (vermelho)", "H (vermelho)", "G (laranja)"]]
    s.append(_bloco_tabela("Tabela 2. Matriz de notas: linhas, eixo quantitativo; colunas, eixo qualitativo.",
                           _tabela(["Quanti \\ Quali", "Insatisfatório", "Satisfatório", "Robusto"], mat,
                                   [3.2 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm], est), est))

    s.append(Paragraph("3.3 Probabilidade de falha material (Basileia, 1996, Tabela 1)", est["h2"]))
    s.append(Paragraph(
        "A probabilidade P de falha material em um ano é a chance de um modelo sair da zona verde do semáforo de "
        "backtesting (cinco ou mais estouros em 250 dias) quando a cobertura real do VaR de 99% é de 99%, 98% ou "
        "97% (BCBS, 1996; Kupiec, 1995). Associa-se Robusto, Satisfatório e Insatisfatório a essas coberturas, e a "
        "mesma escala vale para os dois eixos da nota.", est["p0"]))
    s.append(Paragraph("P(nível) = Pr[X ≥ 5],   X ~ Binomial(250; 1 − cobertura)", est["eq"]))
    s.append(_bloco_tabela("Tabela 3. Calibração de P.", _tabela(
        ["Nível", "Cobertura", "P calculado", "BCBS (1996), Tabela 1"],
        [[l, c, pct(P[l]), ref] for l, c, ref in zip(R.NIVEIS, ("99%", "98%", "97%"), ("10,8%", "56,1%", "87,2%"))],
        [3.2 * cm, 2.4 * cm, 2.6 * cm, 3.8 * cm], est, alinha_dir=[2]), est))

    s.append(Paragraph("3.4 Magnitude da falha (Basileia, 1996, Tabela 2, e fórmula IRB)", est["h2"]))
    s.append(Paragraph(
        "A magnitude S de cada nível é uma proporção π da severidade máxima do risco, com π igual aos acréscimos ao "
        "multiplicador da Tabela 2 de Basileia para 5, 7 e 10 ou mais estouros, divididos pelo acréscimo máximo "
        "(0,40; 0,65; 1,00). Em mercado, a severidade máxima é a punição máxima do semáforo: acréscimo de 1,00 "
        "sobre o multiplicador 3, isto é, 1/3. Em crédito, é o aumento relativo do capital K da fórmula IRB "
        "(BCBS, 2005; Gordy, 2003; Vasicek, 2002) quando a PD verdadeira é o dobro da estimada pelo modelo, fator "
        "de erro observado no caso das perdas do Chief Investment Office do JPMorgan Chase em 2012 (JPMorgan Chase, "
        "2013; U.S. Senate, 2013).", est["p0"]))
    s.append(Paragraph("S<sub>r</sub>(nível) = π(nível) × S<sub>r,máx</sub>,   S<sub>RM,máx</sub> = 1/3,   "
                       "S<sub>RC,máx</sub> = K(2 PD) / K(PD) − 1", est["eq"]))
    s.append(Paragraph(
        f"Com PD de 1%, LGD de 45% e prazo de 2,5 anos: K(1%) = {pct(k1)}, K(2%) = {pct(k2)} e "
        f"S<sub>RC,máx</sub> = {pct(S['RC']['Insatisfatório'])}.", est["p0"]))
    s.append(_bloco_tabela("Tabela 4. Magnitude da falha por nível.", _tabela(
        ["Nível", "π", "S mercado", "S crédito"],
        [[l, num(pi[l]), pct(S["RM"][l]), pct(S["RC"][l])] for l in R.NIVEIS],
        [3.2 * cm, 2.0 * cm, 2.8 * cm, 2.8 * cm], est, alinha_dir=[1, 2, 3]), est))

    s.append(Paragraph("3.5 Dois canais de erro e harmonização com a matriz de cores", est["h2"]))
    s.append(Paragraph(
        "A severidade-base de cada nota combina dois canais de erro. No canal quantitativo, o modelo erra porque "
        "seu desempenho é fraco; no canal qualitativo, o erro nasce de falhas de dados, implementação, uso ou "
        "governança. Em cada canal, a perda esperada é a probabilidade vezes a magnitude do nível do eixo "
        "correspondente, e a severidade é a média ponderada dos canais, com peso ω do canal qualitativo:",
        est["p0"]))
    s.append(Paragraph("s<sub>r</sub>(nota) = 8% × [P<sub>quanti</sub> S<sub>r,quanti</sub> + ω P<sub>quali</sub> "
                       "S<sub>r,quali</sub>] / (1 + ω),   s<sub>r</sub>(sem validação) = 8% × S<sub>r,máx</sub>",
                       est["eq"]))
    s.append(Paragraph(
        "O peso ω é escolhido para que a ordem das severidades reproduza a matriz de cores. A nota G (quanti "
        "insatisfatório, quali robusto) é laranja, pior que seu espelho C (amarelo), o que exige ω &lt; 1; e melhor "
        f"que F (vermelho), o que exige ω &gt; {num(lo)}. Adota-se o centro do intervalo admissível, "
        f"<b>ω = {num(p0.k_quali)}</b>: uma falha qualitativa pesa 80% de uma falha quantitativa de mesmo grau. O "
        "modelo sem validação recebe o pior caso nos dois canais, conceito alinhado à exigência de que a margem de "
        "conservadorismo seja quantificada pela própria incerteza (EBA, 2017).", est["p"]))

    s.append(Paragraph("3.6 Cauda de perda Beta-PERT e FPR calibrado", est["h2"]))
    s.append(Paragraph(
        "A perda por risco de modelo é tratada como variável limitada e assimétrica à direita, com mínimo na nota A "
        "(risco residual de um modelo correto), moda na nota E (centro da matriz) e máximo no modelo sem validação. "
        f"A distribuição Beta-PERT (Malcolm et al., 1959; Vose, 2008) tem parâmetros α = {num(cd['alpha'])} e "
        f"β = {num(cd['beta'])}. Cada nota recebe o quantil correspondente à sua posição u na escala de "
        "severidade, e o FPR calibrado é esse quantil dividido por 8%:", est["p0"]))
    s.append(Paragraph("u = [s(nota) − s(A)] / [s(sem validação) − s(A)],   FPR<sup>cal</sup> = [a + (b − a) "
                       "B<sup>−1</sup>(u; α, β)] / 8%", est["eq"]))
    s.append(KeepTogether([_fig_cauda(p0, fb), Paragraph(
        "Figura 1. Cauda Beta-PERT de crédito (tier G) e posição de cada nota.", est["cap"])]))
    linhas = [[n, R.GRADE[n][0] if n != "n/a" else "-", R.GRADE[n][1] if n != "n/a" else "-", pct(sev[n]),
               num(fb[n]["u"]), pct(cal["RC"][n]), pct(cal["RM"][n])] for n in sorted(sev, key=sev.get)]
    s.append(_bloco_tabela("Tabela 5. Severidade-base, posição na cauda e FPR calibrado por nota (tier G; mercado "
                           "nível 3).", _tabela(
        ["Nota", "Quanti", "Quali", "Severidade crédito", "u", "FPR cal. crédito", "FPR cal. mercado"], linhas,
        [1.3 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 1.4 * cm, 2.5 * cm, 2.5 * cm], est, alinha_dir=[3, 4, 5, 6]), est,
        "Notas em ordem crescente de severidade. n/a: sem validação ou validação vencida."))

    s.append(Paragraph("3.7 Régua padronizada", est["h2"]))
    s.append(Paragraph(
        "As faixas da régua são as cores da matriz, com a nota A separada das demais verdes (melhor nota) e a nota I "
        "separada das vermelhas (pior nota validada). Em cada risco, o valor da faixa é o maior FPR calibrado de suas "
        "notas, arredondado para cima ao primeiro múltiplo de 0,5 ponto percentual, e cada faixa fica ao menos um "
        "passo acima da anterior. Usar o máximo da faixa garante que nenhuma nota receba FPR abaixo do calibrado: a "
        "simplificação só pode aumentar o capital.", est["p0"]))
    linhas = []
    for f in R.FAIXAS:
        nts = R.NOTAS_FAIXA[f]
        mc, mm = max(cal["RC"][n] for n in nts), max(cal["RM"][n] for n in nts)
        linhas.append([C.NOME_FAIXA[f], pct(mc), pct(rg["RC"][f]), pct(mm), pct(rg["RM"][f])])
    s.append(_bloco_tabela("Tabela 6. Derivação da régua padronizada (tier G; mercado nível 3).", _tabela(
        ["Faixa", "Máx. calibrado crédito", "Régua crédito", "Máx. calibrado mercado", "Régua mercado"], linhas,
        [4.2 * cm, 2.8 * cm, 2.3 * cm, 2.8 * cm, 2.3 * cm], est, alinha_dir=[1, 2, 3, 4]), est))
    s.append(KeepTogether([_fig_regua(cal["RC"], rg), Paragraph(
        "Figura 2. FPR calibrado e régua padronizada de crédito por nota (tier G).", est["cap"])]))

    s.append(Paragraph("3.8 Ajustes de tier e de nível de valor justo", est["h2"]))
    s.append(Paragraph(
        f"Modelos de tier M e de nível 2 têm o FPR multiplicado por {num(lm)} e {num(ln2)}, respectivamente. O valor "
        "padrão de 0,65 é a proporção π(Satisfatório) da Tabela 2 de Basileia. No tier M, ele é confirmado pelo ciclo "
        "de validação: a relevância implícita no prazo (1 ano ÷ 3 anos) multiplicada pela razão do tempo em que um "
        "erro permanece sem detecção, com falhas à taxa η = −ln(1 − P) e detecção na validação seguinte. No nível 2, "
        "o fundamento é a hierarquia de valor justo: insumos observáveis implicam menor incerteza de modelo que no "
        "nível 3 (CPC 46; Regulamento Delegado (UE) 2016/101).", est["p0"]))
    s.append(Paragraph("λ<sub>M</sub><sup>ciclo</sup> = (T<sub>G</sub>/T<sub>M</sub>) × f(T<sub>M</sub>) / "
                       "f(T<sub>G</sub>),   f(T) = 1 − (1 − e<sup>−ηT</sup>) / (ηT)", est["eq"]))
    s.append(_bloco_tabela("Tabela 7. Fator de tier M pelo ciclo de validação.", _tabela(
        ["Nível qualitativo", "Tempo com erro, tier G", "Tempo com erro, tier M", "λ M pelo ciclo"],
        [[l, pct(v["f_G"]), pct(v["f_M"]), num(v["lambda"])] for l, v in ciclo.items()],
        [3.4 * cm, 3.4 * cm, 3.4 * cm, 2.8 * cm], est, alinha_dir=[1, 2, 3]), est,
        "Salvaguarda de ordenação com os fatores adotados: " + "; ".join(
            f"{C.NOME_RISCO[r].lower()}, modelo M sem validação {pct(v['M_sem_validacao'])} contra modelo G de nota I "
            f"{pct(v['G_faixa3'])} ({'atendida' if v['ok'] else 'não atendida'})" for r, v in salv.items()) + "."))
    tab_fpr = []
    for nome, (risco, lam) in C.combos(p).items():
        tab_fpr.append([nome] + [pct(lam * rg[risco][f]) for f in R.FAIXAS])
    s.append(_bloco_tabela("Tabela 8. FPR de risco de modelo por faixa e situação de tier e nível.", _tabela(
        ["Situação", "A", "B, D", "C, E", "G", "F, H", "I", "n/a"], tab_fpr,
        [4.4 * cm] + [1.6 * cm] * 7, est, alinha_dir=list(range(1, 8))), est))

    # ------------------------------------------------------------------ 4. execução
    s.append(Paragraph("4. Execução do cálculo sobre a base de entrada", est["h1"]))
    s.append(Paragraph("4.1 Situação de validação na data-base", est["h2"]))
    s.append(Paragraph(
        f"Na data-base de {_data(data_base)}, {rs['escopo_validos']} dos {rs['escopo_total']} modelos do escopo estão "
        "com validação vigente. Para cada modelo, verifica-se se a última validação ocorreu dentro do prazo do tier; "
        "em caso negativo, ou na ausência de validação, a nota efetiva é n/a e o modelo recebe o FPR de pior caso.",
        est["p0"]))
    linhas = [[r["ID"], r["Tier"], r["Nota informada"], _data(r["Última validação"]), _data(r["Válida até"]),
               r["Situação"], r["Nota efetiva"], C.NOME_FAIXA[r["Faixa"]]] for _, r in det.iterrows()]
    s.append(_bloco_tabela("Tabela 9. Situação da validação e faixa da régua por modelo.", _tabela(
        ["ID", "Tier", "Nota", "Última validação", "Válida até", "Situação", "Nota efetiva", "Faixa"], linhas,
        [1.3 * cm, 1.0 * cm, 1.1 * cm, 2.3 * cm, 2.1 * cm, 2.3 * cm, 1.6 * cm, 3.6 * cm], est), est))

    s.append(Paragraph("4.2 FPR, RWA e capital por modelo", est["h2"]))
    linhas = [[r["ID"], r["Carteira ou família"], pct(r["FPR da faixa"]), num(r["λ tier"]), num(r["λ nível"]),
               pct(r["FPR de risco de modelo"]), num(r["Exposição (R$ mi)"]), num(r["RWA (R$ mi)"]),
               num(r["Capital (R$ mi)"])] for _, r in det.iterrows()]
    linhas.append(["Total", "", "", "", "", pct(bruto["fpr_medio"]), num(bruto["saldo"]), num(bruto["rwa"]),
                   num(bruto["addon"])])
    s.append(_bloco_tabela("Tabela 10. Cálculo do capital por modelo (R$ milhões).", _tabela(
        ["ID", "Carteira ou família", "FPR faixa", "λ tier", "λ nível", "FPR", "Exposição", "RWA", "Capital"], linhas,
        [1.3 * cm, 3.3 * cm, 1.6 * cm, 1.2 * cm, 1.2 * cm, 1.6 * cm, 2.1 * cm, 2.0 * cm, 1.7 * cm], est,
        alinha_dir=[2, 3, 4, 5, 6, 7, 8], total=True), est,
        "FPR = FPR da faixa × λ tier × λ nível. RWA = FPR × exposição. Capital = 8% × RWA. No total, o FPR é o "
        "médio ponderado pela exposição."))

    s.append(Paragraph("4.3 Exemplos passo a passo", est["h2"]))
    exemplos = []
    if n_mod:
        exemplos.append(det.loc[det["Capital (R$ mi)"].idxmax()])
        vig = det[det["Situação"] == "Vigente"]
        if len(vig):
            e2 = vig.loc[vig["Capital (R$ mi)"].idxmax()]
            if e2["ID"] != exemplos[0]["ID"]:
                exemplos.append(e2)
        merc = det[(det["Risco"] == "Mercado") & (~det["ID"].isin([e["ID"] for e in exemplos]))]
        if len(merc):
            exemplos.append(merc.loc[merc["Capital (R$ mi)"].idxmax()])
    for e in exemplos:
        sit = ("validação vigente, nota " + e["Nota efetiva"]) if e["Situação"] == "Vigente" else \
            ("sem validação" if e["Situação"] == C.SEM_VAL else "validação vencida")
        niv = f", nível {e['Nível']}" if e["Risco"] == "Mercado" else ""
        s.append(Paragraph(
            f"<b>{e['ID']}, {e['Carteira ou família']}</b> ({e['Risco'].lower()}, tier {e['Tier']}{niv}, {sit}). "
            f"Faixa: {C.NOME_FAIXA[e['Faixa']]}, com FPR de {pct(e['FPR da faixa'])}. "
            f"FPR ajustado = {pct(e['FPR da faixa'])} × {num(e['λ tier'])} × {num(e['λ nível'])} = "
            f"{pct(e['FPR de risco de modelo'])}. RWA = {pct(e['FPR de risco de modelo'])} × "
            f"{num(e['Exposição (R$ mi)'])} = {num(e['RWA (R$ mi)'])}. Capital = 8% × {num(e['RWA (R$ mi)'])} = "
            f"<b>{num(e['Capital (R$ mi)'])}</b>.", est["p0"]))

    s.append(Paragraph("4.4 Consolidação", est["h2"]))
    linhas = [[r["Risco"], str(r["Modelos"]), num(r["Exposição (R$ mi)"]), pct(r["Cobertura de validação"]),
               num(r["RWA (R$ mi)"]), pct(r["FPR médio"]), num(r["Capital (R$ mi)"])] for _, r in resumo.iterrows()]
    s.append(_bloco_tabela("Tabela 11. Capital econômico para risco de modelo consolidado (R$ milhões).", _tabela(
        ["Risco", "Modelos", "Exposição", "Cobertura", "RWA", "FPR médio", "Capital"], linhas,
        [2.2 * cm, 1.6 * cm, 2.8 * cm, 2.2 * cm, 2.6 * cm, 2.0 * cm, 2.4 * cm], est,
        alinha_dir=[1, 2, 3, 4, 5, 6], total=True), est,
        "Cobertura: parcela da exposição atendida por modelos com validação vigente."))
    s.append(Paragraph(
        f"O capital econômico requerido para risco de modelo é de <b>R$ {num(bruto['addon'])} milhões</b>, sobre RWA de "
        f"R$ {num(bruto['rwa'])} milhões. Modelos sem validação vigente respondem por "
        f"{pct(bruto['parcela_nao_validados'])} do capital.", est["p"]))

    s.append(Paragraph("4.5 Cenários", est["h2"]))
    s.append(Paragraph(
        "A Tabela 12 mostra o capital se todos os modelos, ou apenas os pendentes (sem validação ou com validação "
        "vencida), fossem validados na data-base com cada nota, e o capital se nenhum modelo estivesse validado.",
        est["p0"]))
    linhas = [[r["Nota"], num(r["Todos validados com a nota"]), num(r["Só os pendentes validados com a nota"])]
              for _, r in cen.iterrows()]
    linhas += [["Nenhum validado", num(teto), num(teto)], ["Base de entrada", num(atual), num(atual)]]
    s.append(_bloco_tabela("Tabela 12. Capital sob cenários de validação (R$ milhões).", _tabela(
        ["Nota", "Todos validados com a nota", "Só os pendentes validados"], linhas,
        [3.4 * cm, 4.4 * cm, 4.4 * cm], est, alinha_dir=[1, 2]), est))

    # ------------------------------------------------------------------ 5. premissa e RAS
    s.append(Paragraph("5. Premissa prudencial do escopo e apetite a riscos", est["h1"]))
    s.append(Paragraph(
        "O capital de risco de modelo e a declaração de apetite a riscos (RAS) medem populações diferentes. O capital "
        "cobre a fatia material do inventário; a RAS mede a cobertura de validação do inventário total, com um mínimo "
        "definido pela instituição, e a tolerância abaixo de 100% recai sobre os modelos de menor materialidade, que "
        "não entram no capital. Como premissa prudencial, todos os modelos do escopo do capital devem estar com "
        "validação ativa e em dia. A metodologia é geral: se um modelo do escopo estiver sem validação ou vencido, "
        "recebe o FPR de pior caso, que funciona como penalidade por descumprimento da premissa e não como custo de "
        "uma tolerância aceita.", est["p"]))
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
    s.append(_bloco_tabela("Tabela 13. Premissa do escopo e cobertura da RAS.", _tabela(
        ["Indicador", "Valor"], linhas, [8.0 * cm, 5.0 * cm], est, alinha_dir=[1]), est))

    # ------------------------------------------------------------------ 6. limitações
    s.append(Paragraph("6. Governança e limitações", est["h1"]))
    s.append(Paragraph(
        "A régua padronizada é a regra de cálculo e só deve mudar por decisão de comitê. Na revisão anual, "
        "recalcula-se a calibração (P, S, ω, pior caso de crédito, moda da cauda e fatores λ) e verifica-se se alguma "
        "nota passou a ter FPR calibrado acima do valor de sua faixa, se a matriz de cores mudou e se a salvaguarda "
        "de ordenação continua atendida. As principais limitações, incorporadas como premissas, são:", est["p"]))
    for t in ("o capital só varia quando o modelo muda de faixa; mudanças de nota dentro da mesma faixa não são "
              "reconhecidas;",
              "o peso ω e a forma Beta-PERT traduzem a matriz de cores e não são estimados a partir de perdas "
              "observadas por erro de modelo, das quais não há histórico;",
              "o FPR não diferencia carteiras de menor risco de crédito; todas pesam pela exposição, e o risco de "
              "crédito permanece no Pilar 1;",
              "em mercado, o valor justo absoluto mede tamanho e não sensibilidade; posições pequenas e muito "
              "sensíveis a premissas podem ser subestimadas;",
              "erros reais podem superar o pior caso da régua, como no caso de 2012 citado; a cobertura de validação "
              "e a regra de validação vencida são os controles dessa exposição;",
              "os modelos fora do escopo não recebem capital próprio, por baixa materialidade, e são acompanhados "
              "pela RAS e pelo plano de validação."):
        s.append(Paragraph("• " + t, est["item"]))

    # ------------------------------------------------------------------ referências
    s.append(Paragraph("Referências", est["h1"]))
    refs = [
        "BANCO CENTRAL DO BRASIL. Resolução BCB nº 229, de 12 de maio de 2022. Dispõe sobre os procedimentos para o "
        "cálculo da parcela dos ativos ponderados pelo risco relativa às exposições ao risco de crédito mediante "
        "abordagem padronizada.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>Supervisory framework for the use of \"backtesting\" in "
        "conjunction with the internal models approach to market risk capital requirements</i>. Basel: BIS, 1996.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>International convergence of capital measurement and capital "
        "standards: a revised framework</i>. Basel: BIS, 2004.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>An explanatory note on the Basel II IRB risk weight functions</i>. "
        "Basel: BIS, 2005.",
        "BASEL COMMITTEE ON BANKING SUPERVISION. <i>MAR32: Internal models approach: backtesting and P&amp;L "
        "attribution test requirements</i>. Basel Framework. Basel: BIS, 2019.",
        "COMITÊ DE PRONUNCIAMENTOS CONTÁBEIS. <i>CPC 46: Mensuração do valor justo</i>. Brasília, 2012.",
        "CONSELHO MONETÁRIO NACIONAL. Resolução CMN nº 4.557, de 23 de fevereiro de 2017. Dispõe sobre a estrutura de "
        "gerenciamento de riscos e de capital.",
        "CONSELHO MONETÁRIO NACIONAL. Resolução CMN nº 4.966, de 25 de novembro de 2021. Dispõe sobre conceitos e "
        "critérios contábeis aplicáveis a instrumentos financeiros e à contabilidade de hedge.",
        "DERMAN, E. <i>Model risk</i>. Goldman Sachs Quantitative Strategies Research Notes. New York, 1996.",
        "EUROPEAN BANKING AUTHORITY. <i>Guidelines on PD estimation, LGD estimation and the treatment of defaulted "
        "exposures</i> (EBA/GL/2017/16). London, 2017.",
        "EUROPEAN COMMISSION. Commission Delegated Regulation (EU) 2016/101 supplementing Regulation (EU) No 575/2013 "
        "with regard to regulatory technical standards for prudent valuation. <i>Official Journal of the European "
        "Union</i>, 2016.",
        "FEDERAL RESERVE BOARD; OFFICE OF THE COMPTROLLER OF THE CURRENCY. <i>SR 11-7: Supervisory guidance on model "
        "risk management</i>. Washington, 2011.",
        "GORDY, M. B. A risk-factor model foundation for ratings-based bank capital rules. <i>Journal of Financial "
        "Intermediation</i>, v. 12, n. 3, p. 199-232, 2003.",
        "JPMORGAN CHASE &amp; CO. <i>Report of JPMorgan Chase &amp; Co. Management Task Force regarding 2012 CIO "
        "losses</i>. New York, 2013.",
        "KUPIEC, P. H. Techniques for verifying the accuracy of risk measurement models. <i>The Journal of "
        "Derivatives</i>, v. 3, n. 2, p. 73-84, 1995.",
        "MALCOLM, D. G.; ROSEBOOM, J. H.; CLARK, C. E.; FAZAR, W. Application of a technique for research and "
        "development program evaluation. <i>Operations Research</i>, v. 7, n. 5, p. 646-669, 1959.",
        "PRUDENTIAL REGULATION AUTHORITY. <i>SS1/23: Model risk management principles for banks</i>. London: Bank of "
        "England, 2023.",
        "U.S. SENATE. Permanent Subcommittee on Investigations. <i>JPMorgan Chase whale trades: a case history of "
        "derivatives risks and abuses</i>. Washington, 2013.",
        "VASICEK, O. The distribution of loan portfolio value. <i>Risk</i>, v. 15, n. 12, p. 160-162, 2002.",
        "VOSE, D. <i>Risk analysis: a quantitative guide</i>. 3. ed. Chichester: Wiley, 2008.",
    ]
    for r_ in refs:
        s.append(Paragraph(r_, est["ref"]))

    buf = io.BytesIO()
    titulo = "Capital econômico para risco de modelo: memorial de cálculo"
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.4 * cm, rightMargin=2.4 * cm, topMargin=2.2 * cm,
                            bottomMargin=2.2 * cm, title=titulo, author="Memorial gerado pela ferramenta de capital "
                            "de risco de modelo", subject="Capital econômico de Pilar 2 para risco de modelo")
    doc.build(s, onFirstPage=lambda c, d: _rodape(c, d, titulo), onLaterPages=lambda c, d: _rodape(c, d, titulo))
    return buf.getvalue()
