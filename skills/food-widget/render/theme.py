"""Themes and font loading for widget rendering.

The card background carries alpha (#RRGGBBAA) so the wallpaper shows
through. Everything drawn ON the card must use SOLID colors: ImageDraw
replaces pixels (no compositing), so a translucent interior element would
punch a hole in the card. Derived interior tones are pre-blended against
the solid base via blend().
"""
from functools import lru_cache
from pathlib import Path

import matplotlib
from PIL import ImageFont


def _hex(c: str):
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def blend(base: str, over: str, alpha: float) -> str:
    b, o = _hex(base), _hex(over)
    return "#%02X%02X%02X" % tuple(
        round(b[i] + (o[i] - b[i]) * alpha) for i in range(3))


_DARK_BASE = "#10161E"
_LIGHT_BASE = "#FFFFFF"

DARK = {
    "base": _DARK_BASE,
    "bg": _DARK_BASE + "E6",            # translucent card (~90%)
    "border": blend(_DARK_BASE, "#FFFFFF", 0.10),
    "text": "#F2F5F9",
    "sub": "#94A1B5",
    "faint": "#5D6979",
    "bar_bg": blend(_DARK_BASE, "#FFFFFF", 0.13),
    "hairline": blend(_DARK_BASE, "#FFFFFF", 0.08),
    "good": "#34D399",
    "warn": "#FBBF24",
    "bad": "#F87171",
    "accent": "#4DABF7",
    "zones": ["#8B9DB8", "#4DABF7", "#51CF66", "#FFA94D", "#FF6B6B"],
}

LIGHT = {
    "base": _LIGHT_BASE,
    "bg": _LIGHT_BASE + "EB",
    "border": blend(_LIGHT_BASE, "#000000", 0.10),
    "text": "#111827",
    "sub": "#5F6B7C",
    "faint": "#9AA4B2",
    "bar_bg": blend(_LIGHT_BASE, "#000000", 0.10),
    "hairline": blend(_LIGHT_BASE, "#000000", 0.08),
    "good": "#059669",
    "warn": "#D97706",
    "bad": "#DC2626",
    "accent": "#1C7ED6",
    "zones": ["#748199", "#2B8CE6", "#37B24D", "#F08C00", "#F03E3E"],
}

THEMES = {"dark": DARK, "light": LIGHT}

_FONT_DIRS = [
    Path(__file__).resolve().parent.parent / "fonts",
    Path("/usr/share/fonts"),
]

# weight -> (static filename fragments, variable-font instance name)
_WEIGHTS = {
    "regular": (["Regular"], "Regular"),
    "medium": (["Medium"], "Medium"),
    "semibold": (["SemiBold", "DemiBold"], "SemiBold"),
    "bold": (["Bold"], "Bold"),
}


@lru_cache(maxsize=None)
def _resolve(weight: str):
    """Return (font_path, variation_name_or_None) for a weight."""
    fragments, var_name = _WEIGHTS[weight]
    for base in _FONT_DIRS:
        if not base.exists():
            continue
        files = [p for p in base.rglob("Inter*") if p.suffix.lower() in (".ttf", ".otf")]
        for frag in fragments:
            for p in sorted(files):
                if p.stem in (f"Inter-{frag}", f"InterDisplay-{frag}", f"Inter{frag}"):
                    return str(p), None
        for p in sorted(files):
            if "Variable" in p.stem or "[" in p.stem:
                return str(p), var_name
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if weight in ("bold", "semibold"):
        return str(ttf / "DejaVuSans-Bold.ttf"), None
    return str(ttf / "DejaVuSans.ttf"), None


@lru_cache(maxsize=128)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path, variation = _resolve(weight)
    f = ImageFont.truetype(path, size)
    if variation:
        try:
            f.set_variation_by_name(variation)
        except Exception:
            pass
    return f
