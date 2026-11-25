"""Fórmulas em formato de periódico: LaTeX (mathtext do matplotlib, fonte STIX, compatível com Times) convertido em
contornos vetoriais do reportlab, sem imagem rasterizada."""
from __future__ import annotations

import functools

import matplotlib

matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from reportlab.graphics.shapes import FILL_NON_ZERO, Drawing, Group, Path  # noqa: E402
from reportlab.lib import colors  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "stix"
_PROP = FontProperties(family="STIXGeneral")


@functools.lru_cache(maxsize=512)
def _contornos(tex: str, tam: float):
    tp = TextPath((0, 0), tex, size=tam, prop=_PROP)
    ext = tp.get_extents()
    return tp.vertices.copy(), tp.codes.copy() if tp.codes is not None else None, (ext.x0, ext.y0, ext.x1, ext.y1)


def desenho(tex: str, tam: float = 11.0) -> Drawing:
    """Drawing do reportlab com a fórmula; `tex` em sintaxe mathtext, entre cifrões."""
    v, c, (x0, y0, x1, y1) = _contornos(tex, tam)
    p = Path(fillColor=colors.black, strokeColor=None, strokeWidth=0, fillMode=FILL_NON_ZERO)
    i, n = 0, len(v)
    cur = (0.0, 0.0)
    while i < n:
        code = c[i] if c is not None else (MPath.MOVETO if i == 0 else MPath.LINETO)
        x, y = v[i][0] - x0, v[i][1] - y0
        if code == MPath.MOVETO:
            p.moveTo(x, y)
            cur = (x, y)
            i += 1
        elif code == MPath.LINETO:
            p.lineTo(x, y)
            cur = (x, y)
            i += 1
        elif code == MPath.CURVE3:
            qx, qy = x, y
            ex, ey = v[i + 1][0] - x0, v[i + 1][1] - y0
            c1 = (cur[0] + 2 / 3 * (qx - cur[0]), cur[1] + 2 / 3 * (qy - cur[1]))
            c2 = (ex + 2 / 3 * (qx - ex), ey + 2 / 3 * (qy - ey))
            p.curveTo(c1[0], c1[1], c2[0], c2[1], ex, ey)
            cur = (ex, ey)
            i += 2
        elif code == MPath.CURVE4:
            ex, ey = v[i + 2][0] - x0, v[i + 2][1] - y0
            p.curveTo(x, y, v[i + 1][0] - x0, v[i + 1][1] - y0, ex, ey)
            cur = (ex, ey)
            i += 3
        elif code == MPath.CLOSEPOLY:
            p.closePath()
            i += 1
        else:
            i += 1
    d = Drawing(x1 - x0, y1 - y0)
    g = Group(p)
    d.add(g)
    return d


# ----------------------------------------------------------------------------- LaTeX de verdade (pdflatex)
import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

from reportlab.platypus import Flowable  # noqa: E402

_PREAMBULO = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{mathptmx}
\pagestyle{empty}
\newcommand{\eqpage}[1]{%
  \setbox0\hbox{$\displaystyle #1$}%
  \pdfpagewidth=\wd0 \pdfpageheight=\dimexpr\ht0+\dp0\relax
  \pdfhorigin=0pt \pdfvorigin=0pt
  \shipout\box0}
\begin{document}
"""
_CACHE: dict[str, tuple] = {}


def latex_disponivel() -> bool:
    return shutil.which("pdflatex") is not None


def _limpa(tex: str) -> str:
    t = tex.strip()
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1]
    return t


class EquacaoLatex(Flowable):
    """Equação composta pelo pdflatex (Times, mathptmx) e embutida como vetor no PDF do reportlab. Sem LaTeX
    disponível, cai para o desenho em mathtext."""

    def __init__(self, tex: str, tam: float = 11.0, larg_max: float | None = None):
        super().__init__()
        self.tex, self.tam, self.larg_max = _limpa(tex), tam, larg_max
        self._x = None

    def _carrega(self):
        if self._x is None:
            preparar([self])
        return self._x

    def wrap(self, aw, ah):
        x = self._carrega()
        w, h = x["w"], x["h"]
        lim = min(aw, self.larg_max or aw)
        self.esc = min(self.tam / 10.0, lim / w) if w else 1.0
        self.width, self.height = w * self.esc, h * self.esc
        return self.width, self.height

    def draw(self):
        x = self._carrega()
        if x.get("desenho") is not None:
            d = x["desenho"]
            self.canv.saveState()
            self.canv.scale(self.esc, self.esc)
            d.drawOn(self.canv, 0, 0)
            self.canv.restoreState()
            return
        from pdfrw.buildxobj import pagexobj
        from pdfrw.toreportlab import makerl
        xobj = pagexobj(x["pagina"])
        nome = makerl(self.canv, xobj)
        self.canv.saveState()
        self.canv.scale(self.esc, self.esc)
        self.canv.doForm(nome)
        self.canv.restoreState()


def _para_mathtext(t: str) -> str:
    """Traduz os comandos só do LaTeX para o subconjunto do mathtext (alternativa sem pdflatex)."""
    import re
    for c in (r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\big", r"\Big"):
        t = t.replace(c, "")
    t = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", t)
    t = t.replace(r"\Pr", r"\mathrm{Pr}").replace(r"\le ", r"\leq ").replace(r"\Longrightarrow", r"\Rightarrow")
    t = t.replace(r"\frac", r"\dfrac").replace(r"\ddfrac", r"\dfrac")
    return t


def preparar(eqs: list) -> None:
    """Compila de uma vez todas as equações pendentes (uma página por equação) e guarda em cache."""
    pend = [e for e in eqs if e._x is None and e.tex not in _CACHE]
    unicas = list(dict.fromkeys(e.tex for e in pend))
    if unicas:
        ok = False
        if latex_disponivel():
            with tempfile.TemporaryDirectory() as tmp:
                src = _PREAMBULO + "\n".join(r"\eqpage{" + t + "}" for t in unicas) + "\n\\end{document}\n"
                with open(os.path.join(tmp, "eq.tex"), "w", encoding="utf-8") as fh:
                    fh.write(src)
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"], cwd=tmp,
                                   capture_output=True, timeout=120)
                pdf = os.path.join(tmp, "eq.pdf")
                if r.returncode == 0 and os.path.exists(pdf):
                    from pdfrw import PdfReader
                    paginas = PdfReader(pdf).pages
                    if len(paginas) == len(unicas):
                        for t, pg in zip(unicas, paginas):
                            mb = [float(v) for v in pg.MediaBox]
                            _CACHE[t] = {"pagina": pg, "w": mb[2] - mb[0], "h": mb[3] - mb[1]}
                        ok = True
        if not ok:
            for t in unicas:
                d = desenho("$" + _para_mathtext(t) + "$", 10.0)
                _CACHE[t] = {"desenho": d, "w": d.width, "h": d.height}
    for e in eqs:
        if e._x is None:
            e._x = _CACHE.get(e.tex)
