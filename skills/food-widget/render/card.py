"""Shared drawing primitives for widget cards."""
from PIL import Image, ImageDraw

from .theme import font

NARROW_SPACE = " "


def new_card(w: int, h: int, theme: dict):
    """Transparent canvas with a rounded (possibly translucent) card."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    radius = int(min(w, h) * 0.09)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                        fill=theme["bg"], outline=theme["border"], width=2)
    return img, d


def kcal_fmt(n) -> str:
    """1830 -> '1 830' (narrow no-break space)."""
    return f"{int(round(n)):,}".replace(",", NARROW_SPACE)


def pill_bar(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
             frac: float, fill: str, theme: dict, ticks=()):
    """Rounded progress bar with tick separators (fractions 0..1)."""
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=theme["bar_bg"])
    frac = max(0.0, min(frac, 1.0))
    if frac > 0:
        fw = max(h, int(w * frac))
        d.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=fill)
    for t in ticks:
        if 0.03 < t < 0.97:
            tx = x + int(w * t)
            d.line([tx, y + 3, tx, y + h - 3], fill=theme["base"], width=5)


def stacked_bar(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                parts, colors, theme: dict, gap: int = 4):
    """Horizontal stacked bar; parts = raw weights (zeros allowed)."""
    total = sum(p for p in parts if p and p > 0)
    if total <= 0:
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=theme["bar_bg"])
        return
    cx = x
    n_vis = sum(1 for p in parts if p and p > 0)
    avail = w - gap * (n_vis - 1)
    for p, color in zip(parts, colors):
        if not p or p <= 0:
            continue
        pw = max(h, int(avail * (p / total)))
        d.rounded_rectangle([cx, y, min(cx + pw, x + w), y + h],
                            radius=h // 2, fill=color)
        cx += pw + gap
        if cx >= x + w:
            break


def truncate(d: ImageDraw.ImageDraw, s: str, f, max_w: int) -> str:
    """Cut a string with an ellipsis so it fits max_w pixels."""
    if d.textlength(s, font=f) <= max_w:
        return s
    while s and d.textlength(s + "…", font=f) > max_w:
        s = s[:-1]
    return (s + "…") if s else ""


def fit_font(d: ImageDraw.ImageDraw, s: str, size: int, weight: str,
             max_w: int, min_size: int = 40):
    """Largest font of the given weight (<= size) that fits max_w."""
    while size > min_size and d.textlength(s, font=font(size, weight)) > max_w:
        size = int(size * 0.94)
    return font(size, weight), size
