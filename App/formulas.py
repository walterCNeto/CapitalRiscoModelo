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
